import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse, softmax
from typing import List
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add

def pairwise_cosine_single(h, mask):
    """
    h: (L, d)  single sequence (no batch)
    mask: (L,)  1=valid, 0=pad
    returns sim: (L, L) in [-1, 1], pads zeroed out
    """
    valid_idx = mask.nonzero(as_tuple=True)[0]
    h = h[valid_idx]  # (n, d)
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    return sim

def _threshold_edges(sim, tau):
    """
    Binary threshold: edges where sim >= tau, undirected, no self loops.
    """
    A = (sim > tau).float()
    edge_index, edge_weights = dense_to_sparse(A)
    return edge_index, edge_weights

@torch.no_grad()
def build_pyg_graphs(
    hidden,
    attention_mask,
    adjacency="knn",
    tau=0.3,
    device=None,
):
    """
    Convert a batch of token sequences into a list of torch_geometric.data.Data graphs.

    Args
    -----
    hidden: (B, L, d) last-layer token features
    attention_mask: (B, L) 1=valid, 0=pad
    adjacency: "knn" | "threshold" | "soft" | "softmax"
    k, tau, temperature: graph hyper-params
    include_edge_weight: if True and a weighted adjacency is used, store as 'edge_weight'
    device: move Data.x, edge_index, edge_weight to this device (defaults to hidden.device)

    Returns
    -------
    graphs: List[Data], each with fields:
        x: (n, d) token features (valid tokens only)
        edge_index: (2, E)
        edge_weight: (E,) optional for weighted graphs
        num_nodes: n
        token_idx: (n,) original positions within the L-length sequence
    """
    assert hidden.dim() == 3 and attention_mask.dim() == 2, "Bad input shapes"
    B, L, d = hidden.shape
    device = device or hidden.device
    graphs: List[Data] = []

    for b in range(B):
        mask_b = attention_mask[b].to(dtype=torch.bool)
        x_b = hidden[b, mask_b]  # (n, d)
        token_idx = torch.arange(L, device=device)[mask_b]  # (n,)
        n = x_b.size(0)

        sim = pairwise_cosine_single(x_b, mask_b)  # (n, n)

        if adjacency == "threshold":
            edge_index, edge_weight = _threshold_edges(sim, tau=tau)
            data = Data(x=x_b, edge_index=edge_index, edge_attr=edge_weight).to(device)
        else:
            raise ValueError(f"Unknown adjacency: {adjacency}")

        data.token_idx = token_idx
        graphs.append(data)

    return Batch.from_data_list(graphs)

def masked_mean(x, mask, dim):
    mask = mask.to(x.dtype)
    s = (x * mask.unsqueeze(-1)).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1e-6).unsqueeze(-1)
    return s / denom

def masked_max(x, mask, dim):
    # Set masked positions to very small before max
    very_small = torch.finfo(x.dtype).min
    mask_exp = mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * mask_exp + (1 - mask_exp) * very_small
    return x_masked.max(dim=dim).values

class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden, attention_mask):
        return masked_mean(hidden, attention_mask, dim=1)

class MaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden, attention_mask):
        return masked_max(hidden, attention_mask, dim=1)

class CLSPooler(nn.Module):
    def __init__(self, use_last_token_for_decoder=True):
        super().__init__()
        self.use_last_token_for_decoder = use_last_token_for_decoder

    def forward(self, hidden, attention_mask, is_decoder):
        # hidden: (B, L, d)
        if is_decoder and self.use_last_token_for_decoder:
            # pick last non-pad token
            lengths = attention_mask.sum(dim=1)  # (B,)
            idx = (lengths - 1).clamp_min(0).long()
            b_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[b_idx, idx]
        else:
            # first token
            return hidden[:, 0, :]

class MLPPool(nn.Module):
    def __init__(self, inp_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        channel_list = [inp_dim] + [hidden_dim] * num_layers
        self.mlp = MLP(
            channel_list=channel_list
        )
    
    def forward(self, hidden, attention_mask):
        device = hidden.device
        data = build_pyg_graphs(
            hidden, attention_mask, device=device
        )

        data = data.to(device)
        out = self.mlp(data.x, data.batch)
        scores = torch.ones(out.shape[0], device=out.device, dtype=torch.float32).squeeze(-1)
        weights = softmax(scores, data.batch)
        pooled = scatter_add(weights.unsqueeze(-1) * out, data.batch, dim=0)
        return pooled
            
class AdaPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, hidden_states, mask):
        scores = self.score_layer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)
        return pooled
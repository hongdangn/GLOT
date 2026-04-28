import os
import json
import time
import random
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from transformers import AutoTokenizer, AutoConfig, AutoModel
import mteb
from torch_geometric.nn import GATConv, MLP, GINConv, GINEConv, GCNConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from pooling_modules import build_pyg_graphs, AdaPool, MeanPooler, MaxPooler, CLSPooler, MLPPool
from trainer import train_pair_classification, train_pair_embedding, train_single_classification, train_sts_regression, Backbone, \
                    pool_hidden, forward_hidden
from utils import CustomMTEBModel
from data_loader import *
HF_TOKEN = "<>" # Place your huggingface token here

class Encoder:
    def __init__(self, model, tokenizer, pooler, pooler_name, is_decoder, device, batch_size=32, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler
        self.pooler_name = pooler_name
        self.is_decoder = is_decoder
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.model.model = torch.compile(self.model.model)
        self.pooler = torch.compile(self.pooler)

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self.pooler.eval()
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Encoding sentences"):
            batch_sents = sentences[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            hidden, mask = forward_hidden(self.model, inputs)
            pooled = pool_hidden(self.pooler, hidden, mask, self.is_decoder, self.pooler_name)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_decoder_like(config):
    # Heuristic detection of decoder-only causal models
    if getattr(config, "is_decoder", False):
        return True
    mt = getattr(config, "model_type", "") or ""
    # common causal families
    if mt in {"gpt2", "gptj", "gpt_neo", "llama", "mpt", "gemma", "falcon"}:
        return True
    # architectures hint
    arch = getattr(config, "architectures", None)
    if arch:
        if any(("CausalLM" in a) for a in arch):
            return True
    return False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GLOT(nn.Module):
    """
    A pooling head that:
      (i) builds PyG graphs from token features,
      (ii) applies ANY chosen PyG conv stack with Jumping Knowledge,
      (iii) pools tokens with an adaptive scorer.
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_layers=2,
        jk_mode="cat",  
        conv="gat",
        adjacency="threshold",
        tau=0.3,
        use_edge_weight=True,
        device=None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.adjacency = adjacency
        self.tau = tau
        self.use_edge_weight = use_edge_weight
        self.device = device
        
        # Build conv stack
        self.convs = nn.ModuleList()
        last_dim = in_dim
        for _ in range(num_layers):
            if conv == "gat":
                layer = GATConv(last_dim, hidden_dim, edge_dim=1)
            elif conv == "gcn":
                layer = GCNConv(last_dim, hidden_dim)
            elif conv == "gine":
                mlp = MLP([last_dim, hidden_dim, hidden_dim])
                layer = GINEConv(nn=mlp, train_eps=False, edge_dim=1)
            elif conv == "gin":
                mlp = MLP([last_dim, hidden_dim, hidden_dim])
                layer = GINConv(nn=mlp, train_eps=False)
            self.convs.append(layer)
            last_dim = hidden_dim

        if jk_mode == "cat":
            self.out_dim = in_dim + num_layers * hidden_dim
        else:
            self.out_dim = hidden_dim
        
        self.score_layer = nn.Sequential(
            nn.Linear(self.out_dim, max(128, self.out_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(128, self.out_dim // 2), 1)
        )

    def forward(self, hidden, attention_mask):
        """
        hidden: (B, L, d)  token features
        attention_mask: (B, L)  1=valid, 0=pad

        Returns:
          z: (B, D) pooled embeddings
        """
        device = self.device or hidden.device
        B, L, d = hidden.shape
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        #     with record_function("graph_construction"):
        batch = build_pyg_graphs(
            hidden, attention_mask, adjacency=self.adjacency,
            tau=self.tau, device=device, 
        )

        batch = batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        edge_weight = getattr(batch, "edge_attr", None)

        h_list = [x]
        h = x
        # with record_function("gnn"):
        for conv in self.convs:
            if isinstance(conv, GATConv):
                h = conv(h, edge_index, edge_attr=edge_weight)
            elif isinstance(conv, GCNConv):
                h = conv(h, edge_index, edge_weight=edge_weight.squeeze())
            elif isinstance(conv, GINConv):
                h = conv(h, edge_index)
            elif isinstance(conv, GINEConv):
                h = conv(h, edge_index, edge_attr=edge_weight)
            h = F.relu(h)
            h_list.append(h)

        if self.jk_mode == "cat":
            h_all = torch.cat(h_list, dim=-1)
        elif self.jk_mode == "max":
            h_all = torch.stack(h_list[1:], dim=-1).max(dim=-1).values
        elif self.jk_mode == "mean":
            h_all = torch.stack(h_list[1:], dim=-1).mean(dim=-1)
        else:
            raise ValueError("Unknown JK mode") 

        # with record_function("readout"):
        scores = self.score_layer(h_all).squeeze(-1)
        weights = softmax(scores, batch.batch)
        pooled = scatter_add(weights.unsqueeze(-1) * h_all, batch.batch, dim=0)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return pooled

def load_backbone(model_name_or_path, max_length, decoder_cls_last_token=None, task="mteb"):
    config = AutoConfig.from_pretrained(model_name_or_path, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=HF_TOKEN, use_fast=False)
    is_dec = is_decoder_like(config)

    # Ensure padding token & side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if is_dec:
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN, torch_dtype=torch.float16 if task == "mteb" else torch.float32)

    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=64,                       # rank = 64
    #     lora_alpha=64,              # scaling: alpha/r = 1 → no extra scaling
    #     lora_dropout=0.1,
    #     target_modules=["query", "key", "value", "dense"],  # BERT linear layers
    #     bias="none",
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    # model = model.model
    # Ensure model resized if new pad token added
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    # For speed, we do not output hidden_states other than last
    model.eval()

    # Determine CLS strategy
    if decoder_cls_last_token is None:
        decoder_cls_last_token = is_dec

    return Backbone(
        tokenizer=tokenizer,
        model=model, # for lora
        config=config,
        is_decoder=is_dec,
        pad_token_id=tokenizer.pad_token_id,
        model_name_or_path=model_name_or_path
    ), decoder_cls_last_token

# -------------------------
# Encoders & Pooling factory
# -------------------------

def build_pooler(name: str, hidden_size: int, args) -> nn.Module:
    name = name.lower()
    if name == "mean":
        return MeanPooler()
    elif name == "max":
        return MaxPooler()
    elif name == "cls":
        return CLSPooler(use_last_token_for_decoder=args.decoder_cls_last_token)
    elif name == "adapool":
        return AdaPool(in_dim=hidden_size, hidden_dim=args.scorer_hidden)
    elif name == "mlp":
        return MLPPool(inp_dim=hidden_size, hidden_dim=args.gat_hidden_dim, num_layers=args.num_layers)
    elif name == "glot":
        return GLOT(
            in_dim=hidden_size,
            hidden_dim=args.gat_hidden_dim,
            num_layers=args.num_layers,
            jk_mode=args.jk_mode,
            conv=args.gnn_type,
            adjacency=args.graph_adj,
            tau=args.tau,
        )
    else:
        raise ValueError(f"Unknown pooling method: {name}")

def encode_texts(tokenizer: AutoTokenizer, texts, max_length, device):
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in batch.items()}

def evaluate_mteb(
    backbone: Backbone,
    pooler,
    pooler_name,
    device,
    args
):
    # run = wandb.init(project="GLOT")
    # wandb.config.update(args)
    if args.checkpoint_path != "standard" and args.checkpoint_path != "":
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        else:
            checkpoint = torch.load(args.checkpoint_path, weights_only=True, map_location="cpu")
        pooler.load_state_dict(checkpoint)

    model = CustomMTEBModel(
        model_name=None,
        revision=None,
        backbone=backbone,
        pooler=pooler,
        pooler_name=pooler_name,
        device=device,
        args=args
    )

    tasks = mteb.get_tasks(tasks=[args.mteb_task], languages=["eng"])
    results = mteb.evaluate(model, tasks=tasks, encode_kwargs={'batch_size': args.batch_size}, overwrite_strategy="always")

    for result in results:
        print(f"{result.task_name} | {result.get_score()}")
        # wandb.log({f"{result.task_name}": result.get_score()})

    # run.finish()

def run_tasks(backbone: Backbone, args, device):
    pooling_name = args.pooling_method

    task = args.task
    pooler = build_pooler(pooling_name, backbone.config.hidden_size, args).to(device)
    start = time.time()
    summary = {
        "model": args.model_name_or_path,
        "pooling": pooling_name,
        "task": task,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    if task == "stsb":
        train_ds, val_ds = load_stsb(task)
        best = train_sts_regression(backbone, pooler, pooling_name, train_ds, val_ds, args, device)
        summary["metrics"] = {"best_val_avg": best}

    elif task in ["qqp", "mrpc", "rte", "wnli"]: 
        if task == "qqp":
            train_ds, val_ds = load_qqp()
        else:
            train_ds, val_ds = load_stsb(task)
        best = train_pair_classification(backbone, pooler, pooling_name, num_labels=2, train_ds=train_ds, val_ds=val_ds, args=args, device=device)
        summary["metrics"] = {"best_acc": best}

    elif task == "mnli":
        train_ds, val_m, val_mm = load_mnli()
        # Train using matched, evaluate on matched and mismatched
        best_m = train_pair_classification(backbone, pooler, pooling_name, num_labels=3,
                                                    train_ds=train_ds, val_ds=val_m, args=args, device=device, val_ds_mm=val_mm)
        summary["metrics"] = {"best_acc_matched": best_m}

    elif task == "sst2":
        train_ds, val_ds = load_sst2()
        train_ds = train_ds.rename_columns({"sentence": "text"})
        val_ds = val_ds.rename_columns({"sentence": "text"})
        best = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)
        summary["metrics"] = {"best_acc": best}

    elif task == "qnli":
        train_ds, val_ds = load_qnli()
        best = train_pair_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)
        summary["metrics"] = {"best_acc": best}

    elif task == "cola":
        train_ds, val_ds = load_cola()
        train_ds = train_ds.rename_columns({"sentence": "text"})
        val_ds = val_ds.rename_columns({"sentence": "text"})
        best = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)
        summary["metrics"] = {"best_acc": best}

    elif task == "imdb":
        train_ds, test_ds = load_imdb()
        best = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=test_ds, args=args, device=device)
        summary["metrics"] = {"best_acc": best}
    
    elif task == "embedding":
        train_ds = load_embedding_dataset(args.train_file, args.num_train_samples)
        best = train_pair_embedding(backbone, pooler, pooling_name, train_ds, args, device)
    
    elif task == "mteb":
        evaluate_mteb(backbone, pooler, pooling_name, device, args)

    else:
        summary["skipped"] = f"Unknown or unsupported task: {task}"
    
    summary["elapsed_sec"] = round(time.time() - start, 2)

    if args.verbose:
        print(json.dumps(summary, indent=2))

def build_argparser():
    p = argparse.ArgumentParser(description="Train & evaluate LM pooling methods (single-file script).")
    # Model / tokenizer
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model name or path")
    p.add_argument("--decoder_cls_last_token", type=int, default=0,
                   help="If True, CLS pooling uses last non-pad token (for decoder-only). Default: auto-detect.")
    # Tasks & data
    p.add_argument("--task", type=str, default="stsb", help="The dataset to run experiments")
    p.add_argument("--train_file", type=str, default="./data/msmarco-triplets.jsonl", help="Download from https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz")
    p.add_argument("--num_train_samples", type=str, default="subset", help="choose from [subset, full]")
    p.add_argument("--checkpoint_path", type=str, default="standard", help="Pooler Checkpoint path to evaluate on MTEB")
    p.add_argument("--mteb_task", type=str, default="SciFact", help="Clustering or Retrieval")
    p.add_argument("--save_dir", type=str, default="./saved_models/", help="Directory to save logs/results.")
    p.add_argument("--max_length", type=int, default=128, help="Max length for texts")
    p.add_argument("--adaptive_length", type=int, default=0, help="To use full sentence length")
    # Training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)
    # Pooling
    p.add_argument("--pooling_method", type=str, default="max",
                   help="[cls, adapool, max, mean, glot]")
    p.add_argument("--gnn_type", default="gat", type=str)
    p.add_argument("--scorer_hidden", type=int, default=256, help="Hidden dim for adaptive scorer/readout.")
    # GraphPoolJK
    p.add_argument("--gat_hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2, help="Number of GAT layers (K=0 reduces to adaptive scorer).")
    p.add_argument("--jk_mode", type=str, default="cat", choices=["cat", "lstm", "max"])
    p.add_argument("--graph_adj", type=str, default="threshold", choices=["threshold"])
    p.add_argument("--tau", type=float, default=0.3, help="Threshold for adjacency or mid-point for sigmoid.")
    # Projection head
    p.add_argument("--proj_dim", type=int, default=256, help="If >0, apply linear projection to this dim before cosine.")
    # Labels
    p.add_argument("--label_scale", type=str, default="0_1", choices=["0_1", "-1_1", "raw"],
                   help="How to scale STS labels before regression.")
    p.add_argument("--precompute_hidden_states", type=int, default=0, help="Precompute hidden states")
    p.add_argument("--override_precompute", type=int, default=0, help="Override precompute")
    p.add_argument("--finetune_backbone", type=int, default=0, help="Valid when precompute is False and backbone should be finetuned")
    return p

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    backbone, dcls = load_backbone(args.model_name_or_path, max_length=args.max_length, decoder_cls_last_token=args.decoder_cls_last_token, task=args.task)
    args.decoder_cls_last_token = dcls
    backbone.model.to(device)

    run_tasks(backbone, args, device)

if __name__ == "__main__":
    main()
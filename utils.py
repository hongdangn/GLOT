import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from typing import List, Dict
import mteb
from datasets import Dataset
from functools import partial
from pooling_modules import CLSPooler
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Backbone:
    tokenizer: AutoTokenizer
    model: AutoModel
    config: AutoConfig
    is_decoder: bool
    pad_token_id: int
    model_name_or_path: str

def pool_hidden(pooler: nn.Module, hidden: torch.Tensor, attention_mask: torch.Tensor, is_decoder: bool, pooler_name: str):
    if isinstance(pooler, CLSPooler):
        return pooler(hidden, attention_mask, is_decoder)
    else:
        return pooler(hidden, attention_mask)

def forward_hidden(backbone: Backbone, batch_inputs):
    with torch.no_grad():
        outputs = backbone.model(**batch_inputs, return_dict=True, output_hidden_states=True)
        if backbone.is_decoder:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs.last_hidden_state

    attention_mask = batch_inputs["attention_mask"]
    return hidden.float(), attention_mask

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embeddings, passage_embeddings):
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Symmetric loss
        loss_query = F.cross_entropy(similarity_matrix, labels)
        loss_passage = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_query + loss_passage) / 2

class PairClassifier(nn.Module):
    """Linear classifier over r = [z1, z2]."""
    def __init__(self, dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(2 * dim, num_labels)

    def forward(self, z1, z2):
        r = torch.cat([z1, z2], dim=-1)
        return self.classifier(r)

class SingleClassifier(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(dim, num_labels)

    def forward(self, z):
        return self.classifier(z)

class CustomMTEBModel(mteb.EncoderProtocol):
    def __init__(self, model_name, revision, backbone, pooler, pooler_name, device, args):
        self.backbone = backbone
        model_name = self.backbone.model_name_or_path
        revision = None
        self.model_name = model_name
        self.pooler = pooler
        self.pooler_name = pooler_name
        self.tokenizer = backbone.tokenizer
        self.device = device
        self.args = args

        self.backbone.model.eval()

    @torch.no_grad
    def encode(
        self,
        inputs: torch.utils.data.DataLoader[mteb.types.BatchedInput],
        *,
        task_metadata: mteb.TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: mteb.types.PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        sentences = [text for batch in inputs for text in batch["text"]]
        total_sentences = len(sentences)
        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)
        
        try:
            first_batch = next(iter(data_loader))
            first_batch = {k: v.to(self.device) for k, v in first_batch.items()}
            hidden, mask = forward_hidden(self.backbone, first_batch)
            first_z = pool_hidden(self.pooler, hidden, mask, self.backbone.is_decoder, self.pooler_name)
            embedding_dim = first_z.shape[-1]
        except StopIteration:
            # Handle case where data_loader is empty
            return np.array([])
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True
        )

        concatenated_embeds_pt = torch.empty(
            (total_sentences, embedding_dim), 
            dtype=torch.float32, 
            device=self.device
        ) 
        current_index = 0

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc="Encoding"):
            batch_dict = {k: v.to(self.device) for k,v in batch_dict.items()}
            hidden, mask = forward_hidden(self.backbone, batch_dict)
            z = pool_hidden(self.pooler, hidden, mask, self.backbone.is_decoder, self.pooler_name)
            z = F.normalize(z, p=2, dim=-1)

            batch_size = z.shape[0]
            concatenated_embeds_pt[current_index : current_index + batch_size] = z
            current_index += batch_size
           
        concatenated_embeds = concatenated_embeds_pt.cpu().numpy()
        
        return concatenated_embeds
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray):
        return embeddings1 @ embeddings2.T
    
    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        # Handle 1D case (single embedding)
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1[None, :]
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2[None, :]

        # Pairwise dot product = cosine (since normalized)
        return np.sum(embeddings1 * embeddings2, axis=1)
    
def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['input_texts'],
                           max_length=512,
                           padding=True,
                           truncation=True)

    return batch_dict

def pearsonr(x, y):
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean())

def _rankdata(a):
    # average ranks for ties
    temp = a.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a), dtype=float)
    # handle ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ranks = ranks + 1.0  # 1-based
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            avg = ranks[idx].mean()
            ranks[idx] = avg
    return ranks

def spearmanr(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearsonr(rx, ry)

def accuracy(preds, labels):
    return float((preds == labels).mean())

def mcc_binary(preds, labels, eps=1e-12):
    """Matthews correlation coefficient for 0/1 labels."""
    preds = preds.astype(int)
    labels = labels.astype(int)
    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / (denom + eps)

def f1_binary(preds, labels):
    # F1 for positive class (label==1)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
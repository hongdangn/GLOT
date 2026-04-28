import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
import json
from pooling_modules import CLSPooler
import numpy as np
from datetime import datetime

from utils import spearmanr, pearsonr, accuracy, f1_binary, mcc_binary, PairClassifier, SingleClassifier, CustomMTEBModel, ContrastiveLoss
from utils import Backbone, pool_hidden, forward_hidden, mteb
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

class BatchedHiddenStateDataset(torch.utils.data.Dataset):
    def __init__(self, batch_dir):
        self.batch_dir = batch_dir
        meta_file = os.path.join(batch_dir, "metadata.json")

        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)

        self.batch_files = self.metadata["batch_files"]
        self.total_batches = self.metadata["total_batches"]
        self.total_samples = self.metadata["total_samples"]
        self.has_b = self.metadata.get("has_b", False)
        self.has_labels = self.metadata.get("has_labels", False)

        # Precompute cumulative sample counts per batch for indexing
        self.batch_sample_counts = []
        self.cumulative_samples = [0]

        for i, batch_file in enumerate(self.batch_files):
            # You could load shape here, but we assume consistent batch sizes except last
            # Alternatively, store sample_count per batch in metadata
            data = torch.load(batch_file, map_location='cpu')
            sample_count = len(data["a_hs"])
            self.batch_sample_counts.append(sample_count)
            self.cumulative_samples.append(self.cumulative_samples[-1] + sample_count)

        self.cumulative_samples = self.cumulative_samples[:-1]  # remove last dummy

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which batch this sample belongs to
        batch_idx = 0
        while batch_idx < len(self.batch_sample_counts) - 1 and idx >= self.cumulative_samples[batch_idx + 1]:
            batch_idx += 1

        # Load batch (you can add caching here if needed)
        batch_data = torch.load(self.batch_files[batch_idx], map_location='cpu')

        # Find position within batch
        local_idx = idx - self.cumulative_samples[batch_idx]

        # Extract tensors for this sample
        item = [
            batch_data["a_hs"][local_idx],
            batch_data["a_ms"][local_idx],
        ]

        if self.has_b:
            item.extend([
                batch_data["b_hs"][local_idx],
                batch_data["b_ms"][local_idx],
            ])

        if self.has_labels:
            item.append(batch_data["labels"][local_idx])

        return tuple(item)
    
@torch.no_grad()
def precompute_hidden_states(backbone: Backbone, loader, dataset_name, split, save_path, override=False):
    batch_dir = os.path.join(
        save_path,
        f"{backbone.model_name_or_path.replace('/', '_')}_{dataset_name.replace('/', '_')}_{split}_batches"
    )
    meta_file = os.path.join(batch_dir, "metadata.json")

    # Check if already precomputed
    if not override and os.path.exists(meta_file):
        print(f"Loading from precomputed batches in {batch_dir}")
        return BatchedHiddenStateDataset(batch_dir)

    os.makedirs(batch_dir, exist_ok=True)

    device = next(backbone.model.parameters()).device
    batch_files = []
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Precomputing {split}")):
        batch_data = {}

        # Move inputs to device
        if "b_input_ids" in batch:
            a_hidden, a_mask = forward_hidden(backbone, {
                "input_ids": batch["a_input_ids"].to(device),
                "attention_mask": batch["a_attention_mask"].to(device)
            })
            b_hidden, b_mask = forward_hidden(backbone, {
                "input_ids": batch["b_input_ids"].to(device),
                "attention_mask": batch["b_attention_mask"].to(device)
            })
            batch_data.update({
                "a_hs": a_hidden.cpu(),
                "a_ms": a_mask.cpu(),
                "b_hs": b_hidden.cpu(),
                "b_ms": b_mask.cpu(),
            })
        else:
            a_hidden, a_mask = forward_hidden(backbone, {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            })
            batch_data.update({
                "a_hs": a_hidden.cpu(),
                "a_ms": a_mask.cpu(),
            })

        if "labels" in batch:
            batch_data["labels"] = batch["labels"].cpu()

        # Save batch
        batch_file = os.path.join(batch_dir, f"batch_{batch_idx:05d}.pt")
        torch.save(batch_data, batch_file)
        batch_files.append(batch_file)
        total_samples += len(next(iter(batch_data.values())))  # get batch size from any tensor

    # Save metadata
    metadata = {
        "total_batches": len(batch_files),
        "total_samples": total_samples,
        "batch_files": batch_files,
        "has_b": "b_hs" in batch_data,
        "has_labels": "labels" in batch_data,
        "a_hs_shape": batch_data["a_hs"].shape[1:],  # without batch dim
        "a_ms_shape": batch_data["a_ms"].shape[1:],
    }
    if "b_hs" in batch_data:
        metadata.update({
            "b_hs_shape": batch_data["b_hs"].shape[1:],
            "b_ms_shape": batch_data["b_ms"].shape[1:],
        })

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(batch_files)} batches to {batch_dir}")
    return BatchedHiddenStateDataset(batch_dir)

# -------------------------
# Heads for training objectives
# -------------------------
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class ProjectionHead(nn.Module):
    """Optional projection before cosine, e.g., identity by default."""
    def __init__(self, in_dim, out_dim, normalize=True):
        super().__init__()
        self.proj = None
        if out_dim is not None:
            self.proj = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.out_dim = in_dim
        self.normalize = normalize

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            z = self.proj(z)
        if self.normalize:
            z = l2_normalize(z)
        return z

def collate_embedding(examples, tokenizer: AutoTokenizer, device, args):
    texts_a = [ex["query"] for ex in examples]
    texts_b = [ex["pos"][0] for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
    }
    return batch

def collate_pairs(examples, tokenizer: AutoTokenizer, device, args):
    texts_a = [ex["text_a"] for ex in examples]
    texts_b = [ex["text_b"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device)
    }
    return batch

def collate_pairs_cls(examples, tokenizer: AutoTokenizer, device, args):
    # labels as int for classification
    texts_a = [ex["text_a"] for ex in examples]
    texts_b = [ex["text_b"] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device)
    }
    return batch

def collate_single(examples, tokenizer: AutoTokenizer, text_key, device, args):
    texts = [ex[text_key] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch = tokenizer(texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return batch
    
def train_sts_regression(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    train_ds,
    val_ds,
    args,
    device
):
    # labels scale: default [0,5] -> scale to [0,1] per paper
    def scale_label(y):
        if args.label_scale == "0_1":
            return y / 5.0
        elif args.label_scale == "-1_1":
            return (y / 2.5) - 1.0
        else:
            return y

    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, "sts", "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, "sts", "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )

    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    proj = ProjectionHead(in_dim=dim, out_dim=dim, normalize=False).to(device)
    
    # Trainable parts: pooler if it has params (adapool or graphpooljk) + projection head
    trainable = []
    for m in [pooler, proj]:
        if any(p.requires_grad for p in m.parameters()):
            trainable += list(m.parameters())
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            if p.requires_grad:
                trainable.append(p)

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # MSE on cosine similarity
    best_val = -1.0
    for epoch in range(args.epochs):
        pooler.train()
        proj.train()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] STS-B Train ep{epoch+1}"):
            optimizer.zero_grad()
            # Encode A
            if args.precompute_hidden_states:
                a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                labels = scale_label(batch[-1].squeeze()).to(device)
            else:
                a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                labels = scale_label(batch["labels"])
            za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
            za = proj(za)
            # Encode B
            if args.precompute_hidden_states:
                b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
            else:
                b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
            zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
            zb = proj(zb)
            # cosine prediction
            yhat = F.cosine_similarity(za, zb)
            loss = F.mse_loss(yhat, labels)
            loss.backward()
            optimizer.step()
            print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        proj.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        gts = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] STS-B Eval ep{epoch+1}"):
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = scale_label(batch[-1].squeeze()).cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = scale_label(batch["labels"].squeeze()).cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                za = proj(za)
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                zb = proj(zb)
                yhat = F.cosine_similarity(za, zb).cpu().numpy()
                preds.append(yhat)
                gts.append(labels)
                # break # TODO:
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        # Rescale back to original [0,5] for metrics consistency (optional)
        if args.label_scale == "0_1":
            preds_raw = preds * 5.0
            gts_raw = gts * 5.0
        elif args.label_scale == "-1_1":
            preds_raw = (preds + 1.0) * 2.5
            gts_raw = (gts + 1.0) * 2.5
        else:
            preds_raw = preds
            gts_raw = gts

        sp = spearmanr(gts_raw, preds_raw)
        pe = pearsonr(gts_raw, preds_raw)

        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} MSE {avg_loss:.4f} Spearman {sp:.4f} Pearson {pe:.4f}", flush=True)
        best_val = max(best_val, (sp + pe) / 2.0)
    
    return best_val

def train_pair_classification(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    num_labels: int,
    train_ds,
    val_ds,
    args,
    device,
    val_ds_mm=None
):
    
    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, args.task, "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
    if val_ds_mm is not None:
        val_loader_mm = torch.utils.data.DataLoader(
            val_ds_mm,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        if args.precompute_hidden_states:
            val_ds_mm = precompute_hidden_states(backbone, val_loader_mm, args.task, "val_mm", "./data/", override=args.override_precompute)
            val_loader_mm = torch.utils.data.DataLoader(
            val_ds_mm,
            batch_size=args.eval_batch_size,
            shuffle=False
        )
    # Determine pooled dim
    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    classifier = PairClassifier(dim=dim, num_labels=num_labels).to(device)
    params = list(classifier.parameters())
    # Include pooler params if any
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            if p.requires_grad:
                params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0

    for epoch in range(args.epochs):
        pooler.train()
        classifier.train()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] PairCls Train ep{epoch+1}"):
            # Encode A
            if args.precompute_hidden_states:
                a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                labels = batch[-1].squeeze().to(device)
            else:
                a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                labels = batch["labels"]
            za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
            # Encode B
            if args.precompute_hidden_states:
                b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
            else:
                b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
            zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
            logits = classifier(za, zb)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        classifier.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] PairCls Eval ep{epoch+1}"):
                # Encode A
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                # Encode B
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                logits = classifier(za, zb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)
                # break
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        f1 = f1_binary(preds_all, labels_all) if num_labels == 2 else float('nan')
        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f} acc {acc:.4f} f1 {f1:.4f}")
        best_acc = max(best_acc, acc)
    
    if val_ds_mm is not None:
        pooler.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader_mm, desc=f"[{pooler_name}] PairCls Eval Mismatched ep{epoch+1}"):
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                logits = classifier(za, zb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)
                
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        f1 = f1_binary(preds_all, labels_all) if num_labels == 2 else float('nan')

    return best_acc

def train_single_classification(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    num_labels: int,
    train_ds,
    val_ds,
    args,
    device
):

    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2,
            collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, args.task, "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2,
            collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )

    # Determine pooled dim
    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        hidden, mask = sample[0].to(device), sample[1].to(device)
    else:
        hidden, mask = forward_hidden(backbone, {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]})
    z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    classifier = SingleClassifier(dim=dim, num_labels=num_labels).to(device)
    params = list(classifier.parameters())
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            p.requires_grad = True
            params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # print(f"Total trainable params = {sum([p.numel() for p in params])}")

    best_acc = 0.0
    for epoch in range(args.epochs):
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        pooler.train()
        classifier.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] SingleCls Train ep{epoch+1}"):
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            if args.precompute_hidden_states:
                hidden, mask = batch[0].to(device), batch[1].to(device)
                labels = batch[-1].squeeze().to(device)
            else:
                hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
                labels = batch["labels"]
            z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
            logits = classifier(z)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            # print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        classifier.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] SingleCls Eval ep{epoch+1}"):
                if args.precompute_hidden_states:
                    hidden, mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
                logits = classifier(z)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        mcc = mcc_binary(preds_all, labels_all)
        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f} acc {acc:.4f} mcc {mcc:.4f}")
        best_acc = max(best_acc, acc)
    return best_acc

def train_pair_embedding(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    train_ds,
    args,
    device,
):
    
    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_embedding(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_embedding(ex, backbone.tokenizer, device, args)
        )
    # Determine pooled dim
    sample = next(iter(train_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    params = []
    # Include pooler params if any
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if len(params) != 0:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=50)
    else:
        optimizer = None

    loss_fn = ContrastiveLoss()
    best_acc = 0.0

    if optimizer is not None:
        for epoch in range(args.epochs):
            pooler.train()
            losses = []
            for batch in tqdm(train_loader, desc=f"[{pooler_name}] PairEmb Train ep{epoch+1}"):
                # Encode A
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                # Encode B
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                
                loss = loss_fn(za, zb)
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                losses.append(loss.item())
                
            avg_loss = float(np.mean(losses)) if losses else 0.0

            # Eval
            if args.verbose:
                print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f}")
            best_acc = min(best_acc, avg_loss)
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
            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(model, overwrite_results=True)
            for result in results:
                print(f"{result.task_name} | {result.get_score()}")

    
    # Create informative filename based on args
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Build config string with key parameters
    config_str = f"{args.task}_{args.model_name_or_path.replace('/', '_')}_{args.pooling_method}"

    # Add key hyperparameters
    if args.pooling_method == "glot":
        config_str += f"_layers{args.num_layers}_{args.jk_mode}"

    config_str += f"_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_len{args.max_length}"

    if args.proj_dim > 0:
        config_str += f"_proj{args.proj_dim}"

    if args.num_train_samples != "full":
        config_str += f"_samples{args.num_train_samples}"

    # Create final path
    save_path = os.path.join(
        args.save_dir, 
        f"{config_str}_{timestamp}.pth"
    )
    if optimizer is not None:
        torch.save(pooler.state_dict(), save_path)
    
    
    return best_acc
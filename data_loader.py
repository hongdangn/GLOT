from datasets import load_dataset

def load_stsb(task):
    # GLUE STS-B 
    # and also task = [mrpc, rte, wnli]
    ds = load_dataset("glue", task)
    train = ds["train"]
    val = ds["validation"]
    # Map to common fields
    train = train.rename_columns({"sentence1": "text_a", "sentence2": "text_b"})
    val = val.rename_columns({"sentence1": "text_a", "sentence2": "text_b"})
    return train, val

def load_qqp():
    # ds = load_dataset("glue", "qqp")
    train = load_dataset("glue", "qqp", split="train[:20000]")
    val = load_dataset("glue", "qqp", split="validation")
    train = train.rename_columns({"question1": "text_a", "question2": "text_b"})
    val = val.rename_columns({"question1": "text_a", "question2": "text_b"})
    return train, val

def load_qnli():
    # ds = load_dataset("glue", "qnli")
    train = load_dataset("glue", "qnli", split="train[:20000]")
    val = load_dataset("glue", "qnli", split="validation")
    train = train.rename_columns({"question": "text_a", "sentence": "text_b"})
    val = val.rename_columns({"question": "text_a", "sentence": "text_b"})
    return train, val

def load_mnli():
    # ds = load_dataset("glue", "mnli")
    train = load_dataset("glue", "mnli", split="train[:20000]")
    val_m = load_dataset("glue", "mnli", split="validation_matched")
    val_mm = load_dataset("glue", "mnli", split="validation_mismatched")
    train = train.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    val_m = val_m.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    val_mm = val_mm.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    return train, val_m, val_mm

def load_sst2():
    ds = load_dataset("glue", "sst2")
    return ds["train"], ds["validation"]

def load_cola():
    ds = load_dataset("glue", "cola")
    return ds["train"], ds["validation"]

def load_imdb():
    ds = load_dataset("imdb")
    return ds["train"], ds["test"]

def load_embedding_dataset(train_file, num_samples):
    if num_samples == "subset":
        split = "train[:20000]"
    else:
        split = "train"
    ds = load_dataset("json", data_files=train_file, split=split)
    return ds
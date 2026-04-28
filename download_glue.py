from datasets import load_dataset

tasks = ["cola", "sst2", "stsb", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]

for task in tasks:
    print(f"Downloading {task}...")
    load_dataset("glue", task)

print("All datasets downloaded successfully!")
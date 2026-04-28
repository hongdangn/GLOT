import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Download incomplete.")
    else:
        print(f"\nDone! File saved as {filename}")

if __name__ == "__main__":
    target_url = "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz"
    download_file(target_url, "msmarco-triplets.jsonl.gz")
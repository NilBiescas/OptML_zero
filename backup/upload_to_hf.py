import os
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        # try reading from .env
        with open(".env") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=")[1].strip('"\'')
    if not token:
        print("Please set HF_TOKEN in .env")
        return

    api = HfApi(token=token)
    
    try:
        username = api.whoami()["name"]
    except Exception as e:
        print(f"Failed to authenticate with HF_TOKEN. Please make sure your token is valid and has WRITE permissions! Error: {e}")
        return

    repo_id = f"{username}/LM-BFF-datasets"
    
    print(f"Creating dataset repo {repo_id} (Private)...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"Repo might already exist: {e}")
        
    print("Uploading the extracted dataset folder to Hugging Face...")
    api.upload_folder(
        folder_path="LOZO/data/original",
        path_in_repo="original",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Upload complete! The cluster pods can now download it instantly.")

if __name__ == "__main__":
    main()

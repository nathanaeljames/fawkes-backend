# A standalone program to download and cache the entire Canary-Qwen model
# from the Hugging Face Hub for offline use.

import os
from huggingface_hub import snapshot_download

def download_model_and_tokenizer(model_name: str, local_dir: str):
    """
    Downloads all files for a specified Hugging Face model to a local directory.

    Args:
        model_name (str): The name of the model on Hugging Face Hub (e.g., "nvidia/canary-qwen-2.5b").
        local_dir (str): The local path where all model files will be saved.
    """
    print(f"Starting download of '{model_name}' to '{local_dir}'...")
    print("This may take some time depending on your internet connection.")
    print("All necessary files, including the model and tokenizer, will be downloaded.")

    try:
        # The snapshot_download function is the best way to download an entire
        # repository, including all model files, tokenizer files, and configs.
        # This will create a directory structure that is compatible with
        # from_pretrained() methods.
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print("\nDownload complete! All model and tokenizer files are now saved locally.")
        print(f"You can now load the model from the local path: {local_dir}")
    except Exception as e:
        print(f"\nAn error occurred during the download: {e}")
        print("Please check the model name and your internet connection.")

if __name__ == "__main__":
    # The Hugging Face model name to download.
    MODEL_NAME = "nvidia/canary-qwen-2.5b"
    
    # The local path where the model files will be saved.
    # This must be the same path you will use in your transcription script.
    LOCAL_PATH = "/root/fawkes/models/canary-qwen-2.5b"
    
    # Run the download function
    download_model_and_tokenizer(MODEL_NAME, LOCAL_PATH)


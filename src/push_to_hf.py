from huggingface_hub import HfApi, HfFolder
from credentials_utils import setup_huggingface_hub
import os

def push_to_hf(model_path, repo_id):
    # Setup HuggingFace 
    setup_huggingface_hub()

    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        raise ValueError("Hugging Face token not found. Ensure you are logged in.")

    # Upload model to the Hub
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model/trashnet_model.pth",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model pushed to Hugging Face: {repo_id}")

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "best_trashnet_model.pth")
    repo_id = "grediiiii/trashnet-adamata"
    push_to_hf(model_path, repo_id)

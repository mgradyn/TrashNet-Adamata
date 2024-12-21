import os
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

if os.getenv("GITHUB_ACTIONS") is None: 
    load_dotenv()

# Access the token
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
wandb_key = os.getenv("WANDB_API_KEY")

if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Ensure it is available in your environment.")
if not wandb_key:
    raise ValueError("WANDB_API_KEY is not set. Ensure it is available in your environment.")

def setup_wandb(project_name, config):
    wandb.login(key=wandb_key)
    wandb.init(project=project_name, config=config)

def setup_huggingface_hub():
    login(token=huggingface_token)
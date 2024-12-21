import os
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

if os.getenv("GITHUB_ACTIONS") is None: 
    load_dotenv()

# Access the token
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
wandb_key = os.getenv("WANDB_API_KEY")

def setup_wandb(project_name, config):
    wandb.login(key=wandb_key)
    wandb.init(project=project_name, config=config)

def setup_huggingface_hub():
    login(token=huggingface_token)
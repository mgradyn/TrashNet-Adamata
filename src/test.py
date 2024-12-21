import os
import json
import torch
from collections import Counter
from test_utils import TestUtils
from train_utils import TrainUtils
from dataset_utils import DatasetUtils
from model import TrashNet
from credentials_utils import setup_huggingface_hub

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def print_metrics(phase, metrics, num_classes):
    print(
        f"\n{phase} Metrics:\n"
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision (Macro): {metrics['precision_macro']:.4f} | "
        f"Recall (Macro): {metrics['recall_macro']:.4f} | "
        f"F1 (Macro): {metrics['f1_macro']:.4f} | "
        f"Precision (Micro): {metrics['precision_micro']:.4f} | "
        f"Recall (Micro): {metrics['recall_micro']:.4f} | "
        f"F1 (Micro): {metrics['f1_micro']:.4f}"
    )
    print(f"\n{phase} Per-Class Metrics:")
    for class_idx in range(num_classes):
        print(
            f"Class {class_idx}: "
            f"Precision: {metrics['per_class']['precision'][class_idx]:.4f} | "
            f"Recall: {metrics['per_class']['recall'][class_idx]:.4f} | "
            f"F1: {metrics['per_class']['f1'][class_idx]:.4f}"
        )   

def main():
    # Setup HuggingFace and WandB
    setup_huggingface_hub()

    # Dataset Preparation
    dataset = DatasetUtils.load_dataset()
    dataset = DatasetUtils.remove_class(dataset, split="train", class_label=5) # remove ambigious class (class 5)
    dataset = DatasetUtils.split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1)

    # create DataLoaders
    _, val_test_transform = DatasetUtils.get_transforms()
    _, _, test_loader = DatasetUtils.create_dataloaders(
        dataset, None, val_test_transform, test_batch_size=BATCH_SIZE
    )

    # Model Setup
    class_counts = Counter(dataset["train"]["label"])
    num_classes = len(class_counts)

    # Load model configuration and weights
    config_path = "./config/model_cfg.json"
    model_path = "./model/best_trashnet_model.pth"
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Required configuration or model file is missing.")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = TrashNet(cfgs=cfg["cfgs"], num_classes=num_classes, width=cfg["width"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    torch.backends.cudnn.benchmark = True

    # Evaluate on the test set
    test_metrics = TestUtils.test_model(model, test_loader, DEVICE, num_classes)

    # Print test metrics
    print_metrics("Test", test_metrics, num_classes)

if __name__ == "__main__":
    main()

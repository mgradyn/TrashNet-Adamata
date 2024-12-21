import os
import torch
import wandb
import json
from collections import Counter
from train_utils import TrainUtils
from dataset_utils import DatasetUtils
from regularization_utils import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TrashNet
from credentials_utils import setup_wandb, setup_huggingface_hub

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def save_best_model(model, optimizer, scheduler, epoch, train_metrics, val_metrics):
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    torch.save(save_dict, "./model/best_trashnet_model.pth")
    print("\nSaved best model.")

def print_metrics(phase, loss, metrics, num_classes):
    print(
        f"\n{phase} Metrics:\n"
        f"Loss: {loss:.4f} | "
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
    setup_wandb(
        project_name="trashnet-training",
        config={
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
    )

    # Dataset Preparation
    dataset = DatasetUtils.load_dataset()
    dataset = DatasetUtils.remove_class(dataset, split="train", class_label=5) # remove ambigious class (class 5)
    dataset = DatasetUtils.split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1)

    # create DataLoaders
    train_transform, val_test_transform = DatasetUtils.get_transforms()
    train_loader, val_loader, _ = DatasetUtils.create_dataloaders(
        dataset, train_transform, val_test_transform, train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE
    )

    # Model Setup
    class_counts = Counter(dataset["train"]["label"])
    num_classes = len(class_counts)

    with open("./config/model_cfg.json", "r") as f:
        cfg = json.load(f)
    model = TrashNet(cfgs=cfg["cfgs"], num_classes=num_classes, width=cfg["width"], dropout=cfg["dropout"]).to(DEVICE)


    torch.backends.cudnn.benchmark = True

    class_weights = TrainUtils.compute_class_weights(class_counts, num_classes, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    early_stopping = EarlyStopping(patience=5, delta=0.001, path="best_trashnet_model.pth")

    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_metrics = TrainUtils.train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, num_classes)

        print_metrics("Training", train_loss, train_metrics, num_classes)
      
        val_loss, val_metrics = TrainUtils.validate_one_epoch(model, val_loader, criterion, DEVICE, num_classes)

        print_metrics("Validation", val_loss, val_metrics, num_classes)

        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_best_model(model, optimizer, scheduler, epoch, train_metrics, val_metrics)
          
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        scheduler.step(val_loss)
        torch.cuda.empty_cache()

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}")
  

if __name__ == "__main__":
    main()

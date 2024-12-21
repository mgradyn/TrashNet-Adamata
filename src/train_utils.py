import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.cuda.amp import autocast

class TrainUtils:
    @staticmethod
    def compute_class_weights(class_counts, num_classes, device):
        class_weights = torch.tensor(
            [1.0 / class_counts.get(i, 1e-6) for i in range(num_classes)],
            dtype=torch.float
        )
        class_weights /= class_weights.sum()
        return class_weights.to(device)

    @staticmethod
    def compute_metrics(y_true, y_pred, num_classes):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Ensure valid inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Accuracy
        accuracy = np.mean(y_true == y_pred)

        # Precision, Recall, and F1 score (per class)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)

        for cls in range(num_classes):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[cls] = (
                2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
                if (precision[cls] + recall[cls]) > 0
                else 0
            )

        # Macro-Averaged Metrics
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)

        # Micro-Averaged Metrics (treat as binary classification across all classes)
        tp_micro = np.sum((y_true == y_pred))
        fp_micro = np.sum((y_pred != y_true) & (y_pred != -1))
        fn_micro = np.sum((y_true != y_pred) & (y_true != -1))

        precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        f1_micro = (
            2 * precision_micro * recall_micro / (precision_micro + recall_micro)
            if (precision_micro + recall_micro) > 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "per_class": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        }

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
            else:
                print(f"Warning: Ignoring out-of-range values t={t}, p={p}")
        return cm


    @staticmethod
    def train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes):
        model.train()
        scaler = torch.amp.GradScaler(device)
        total_loss = 0.0
        accumulation_steps = 4
        y_true, y_pred = [], []

        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Collect predictions and labels
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Cumulative loss
            total_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Compute metrics for every 10 batch
            if (batch_idx + 1) % 10 == 0:
                batch_metrics = compute_metrics(y_true[-len(labels):], preds.cpu().numpy(), num_classes)
                print(f"[Train Batch {batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | "
                      f"Accuracy: {batch_metrics['accuracy']:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Compute epoch-level metrics
        metrics = compute_metrics(y_true, y_pred, num_classes)
        confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)

        # Log metrics to wandb
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_accuracy": metrics["accuracy"],
            "train_precision_macro": metrics["precision_macro"],
            "train_recall_macro": metrics["recall_macro"],
            "train_f1_macro": metrics["f1_macro"],
            "train_precision_micro": metrics["precision_micro"],
            "train_recall_micro": metrics["recall_micro"],
            "train_f1_micro": metrics["f1_micro"],
            "train_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=np.array(y_true), preds=np.array(y_pred), class_names=[str(i) for i in range(num_classes)]
            ),
        })

        return total_loss / len(train_loader), metrics

    def validate_one_epoch(model, val_loader, criterion, device, num_classes):
        model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Collect predictions and labels
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                # Cumulative loss
                total_loss += loss.item()

                # Compute metrics for every 10 batch
                if (batch_idx + 1) % 10 == 0:
                    batch_metrics = compute_metrics(y_true[-len(labels):], preds.cpu().numpy(), num_classes)
                    print(f"[Val Batch {batch_idx + 1}/{len(val_loader)}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Accuracy: {batch_metrics['accuracy']:.4f}")

        # Compute epoch-level metrics
        metrics = compute_metrics(y_true, y_pred, num_classes)
        confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)

        # Log metrics to wandb
        wandb.log({
            "val_loss": total_loss / len(val_loader),
            "val_accuracy": metrics["accuracy"],
            "val_precision_macro": metrics["precision_macro"],
            "val_recall_macro": metrics["recall_macro"],
            "val_f1_macro": metrics["f1_macro"],
            "val_precision_micro": metrics["precision_micro"],
            "val_recall_micro": metrics["recall_micro"],
            "val_f1_micro": metrics["f1_micro"],
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=np.array(y_true), preds=np.array(y_pred), class_names=[str(i) for i in range(num_classes)]
            ),
        })

        return total_loss / len(val_loader), metrics
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TestUtils:
    @staticmethod
    def test_model(model, test_loader, device, num_classes):
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        metrics = compute_metrics(y_true, y_pred, num_classes)
        confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes)

        print("\nTest Results:")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"{k}: {v:.4f}")

        print("\nConfusion Matrix:")

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(num_classes)], yticklabels=[f'Class {i}' for i in range(num_classes)])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        return metrics
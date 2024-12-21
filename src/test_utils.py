import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from train_utils import TrainUtils
import numpy as np

class TestUtils:
    @staticmethod
    def test_model(model, dataset, test_loader, device, num_classes):
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        metrics = TrainUtils.compute_metrics(y_true, y_pred, num_classes)
        confusion_matrix = TrainUtils.compute_confusion_matrix(y_true, y_pred, num_classes)
        label_mapping = dataset['test'].features["label"].int2str

        print("\nTest Results:")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"{k}: {v:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix)

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[label_mapping(i) for i in range(num_classes)], yticklabels=[label_mapping(i) for i in range(num_classes)])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        return metrics
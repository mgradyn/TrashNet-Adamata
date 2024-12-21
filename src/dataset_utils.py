from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np
from torchvision import transforms

# Dataset Loading and Preprocessing Functions
class DatasetUtils:

    @staticmethod
    def load_dataset():
        return load_dataset("garythung/trashnet")

    @staticmethod
    def display_random_images(dataset, num_images=5):
        label_mapping = dataset.features["label"].int2str

        random_indices = random.sample(range(len(dataset)), num_images)
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(random_indices):
            example = dataset[idx]
            img = example["image"]
            label = example["label"]
            label_name = label_mapping(label)

            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(f"Label: {label_name}")
            plt.axis("off")
        plt.show()

    @staticmethod
    def analyze_image_sizes(dataset):
        widths, heights = [], []
        for example in dataset:
            img = example["image"]
            widths.append(img.width)
            heights.append(img.height)

        # Calculate and display statistics
        print(f"Average Width: {sum(widths) / len(widths):.2f}")
        print(f"Average Height: {sum(heights) / len(heights):.2f}")
        print(f"Max Width: {max(widths)}, Min Width: {min(widths)}")
        print(f"Max Height: {max(heights)}, Min Height: {min(heights)}")

        # Plot histograms of widths and heights
        plt.figure(figsize=(10, 5))
        plt.hist(widths, bins=20, alpha=0.5, label="Widths")
        plt.hist(heights, bins=20, alpha=0.5, label="Heights")
        plt.legend()
        plt.title("Image Dimensions Distribution")
        plt.xlabel("Pixels")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def analyze_image_channels(dataset, num_samples=100):
        channel_means = {"R": [], "G": [], "B": []}
        sample_indices = random.sample(range(len(dataset)), num_samples)

        for idx in sample_indices:
            img = np.array(dataset[idx]["image"])
            channel_means["R"].append(np.mean(img[:, :, 0]))
            channel_means["G"].append(np.mean(img[:, :, 1]))
            channel_means["B"].append(np.mean(img[:, :, 2]))

        # Calculate global mean for each channel
        global_means = {channel: np.mean(means) for channel, means in channel_means.items()}
        print(f"Average R Mean: {global_means['R']:.2f}")
        print(f"Average G Mean: {global_means['G']:.2f}")
        print(f"Average B Mean: {global_means['B']:.2f}")

        # Plot histograms for each channel
        plt.figure(figsize=(15, 5))
        for i, (channel, means) in enumerate(channel_means.items()):
            plt.subplot(1, 3, i + 1)
            plt.hist(means, bins=20, color=channel.lower(), alpha=0.7)
            plt.title(f"{channel}-Channel Mean Distribution")
            plt.xlabel("Mean Value")
            plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def plot_class_distribution(dataset):
        labels = dataset["label"]
        label_counts = Counter(labels)

        # Display class frequencies
        print("Class Distribution:")
        for label, count in label_counts.items():
            print(f"Class {label}: {count} images")

        # Plot class distribution
        plt.figure(figsize=(10, 5))
        plt.bar(label_counts.keys(), label_counts.values(), color="skyblue")
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def display_class_images(dataset, class_label=5, num_images=15):
        label_mapping = dataset.features["label"].int2str

        target_indices = []
        threshold = num_images * 3

        for idx in range(len(dataset) - 1, -1, -1):
            if dataset[idx]["label"] == class_label:
                target_indices.append(idx)
            if len(target_indices) >= threshold:
                break

        if len(target_indices) < num_images:
            print("Warning: Not enough images found for the specified class.")
            num_images = len(target_indices)

        random_indices = sample(target_indices, num_images)

        cols = 3
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 5 * rows))

        for i, idx in enumerate(random_indices):
            example = dataset[idx]
            img = example["image"]
            label = example["label"]
            label_name = label_mapping(label)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Image {i+1}, Label: {label_name}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
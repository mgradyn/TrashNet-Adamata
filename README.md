# TrashNet-Adamata

## Overview
TrashNet-Adamata is a lightweight and efficient convolutional neural network (CNN) model designed for classification tasks. The model leverages:
- A combination of traditional convolution and depthwise convolution to extract additional features while reducing model size and computational cost.
- Channel and spatial attention mechanisms integrated into the bottleneck layer to focus on the most critical features.

**Key Model Characteristics:**
- Total parameters: **3.16 million**
- Model size: **12 MB**

The development and training details are further explained in the accompanying Jupyter notebooks.

## Training Results
The model was trained on Google Colab's free T4 GPU for **28 epochs** due to resource constraints. Despite the limited training time, the model achieved the following performance on the testing set:
- **Accuracy**: 0.8145
- **Precision (Macro)**: 0.8217
- **Recall (Macro)**: 0.8155
- **F1 Score (Macro)**: 0.8179

Given more training epochs, the model is expected to achieve even better results, potentially reaching around **90% accuracy**.

## Repository Structure
- **Jupyter Notebooks**
  - `adamata-trashnet-exploratory-analysis.ipynb`: A detailed exploration of the dataset, model, training, and testing processes.
  - `adamata-trashnet-repo-based.ipynb`: A clean, organized version of the exploratory notebook based on the source code provided in this repository. This notebook simplifies reproducibility and demonstrates how to use the utilities for dataset handling, training, and testing.

- **GitHub Actions**
  - **Test**: Automatically triggered on every push. This action tests the model and logs performance metrics to ensure updates do not degrade model performance.
  - **Push to HuggingFace**: A manual action for uploading new models from `src/model` to the HuggingFace repository.
  - **Train**: A manual action that can be triggered but requires a GPU-enabled environment.

## Instructions for Reproducibility
1. Clone the repository.
2. Create a `.env` file in the root directory and add the following:
   ```env
   HUGGINGFACE_TOKEN=your_token
   WANDB_API_KEY=your_key
   ```
3. Run the `adamata-trashnet-repo-based.ipynb` notebook to reproduce results and test the model.

## HuggingFace Repository
The trained model is hosted on HuggingFace. Visit the model repository here: [TrashNet-Adamata on HuggingFace](https://huggingface.co/grediiiii/trashnet-adamata).

## Development Logs
Development logs, including loss, accuracy, precision (macro/micro), recall (macro/micro), F1 scores (macro/micro), and confusion matrix history, can be accessed on [Weights & Biases (Wandb)](https://wandb.ai/mgradyn/trashnet-classification/runs/x74uwf2n?nw=nwusermgradyn).

## Acknowledgments
- Model trained on Google Colab's free T4 GPU.
- Special thanks to open-source libraries like PyTorch, Weights & Biases, and HuggingFace for enabling this work.

---
**Disclaimer:** This model and repository were developed under constrained resources and time, as part of a thesis project, while managing full-time work commitments. Further improvements and updates are anticipated.


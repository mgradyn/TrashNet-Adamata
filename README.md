# AdaMata TrashNet 🌍♻

Welcome to the **AdaMata TrashNet** repository! This project is my exploration into building a lightweight, efficient classification model using PyTorch. Here’s the story behind it and everything you need to know to dive in. Let’s go!

---

## The Model

I designed a convolutional model combining:

1. **Traditional Convolutions** and **Depthwise Convolutions**:

   - Traditional convolutions extract initial features.
   - Depthwise convolutions extract additional features without inflating parameters, making the model lightweight.

2. **Channel and Spatial Attention**:
   - Attention mechanisms in the bottleneck layer focus the model on the most important features.

### Specs:

- **Parameters:** 3.16 million
- **Model Size:** 12MB

Further details about the architecture are in the accompanying Jupyter notebook.

---

## The Training Journey 🚀

Training was done on **Colab T4 (Free)** because... let’s just say my **GTX 1050** wasn’t up to the task. (Also, broke life problems.) With **27 epochs** squeezed in (thanks to Colab’s generous but limited sessions), here’s how it performed on the test set:

- **Accuracy:** 0.8145
- **Precision (Macro):** 0.8217
- **Recall (Macro):** 0.8155
- **F1 (Macro):** 0.8179

Given more training time, I’m confident it could hit 90%+ accuracy. But deadlines are deadlines (thesis week, full-time work, and this… you get it).

---

## Notebooks

Two Jupyter notebooks are included for your convenience:

### 1. `adamata-trashnet-exploratory-analysis.ipynb`

- This notebook covers exploratory data analysis, model design, training, and testing. Think of it as the messy but rich backstory of the project.

### 2. `adamata-trashnet-repo-based.ipynb`

- A streamlined version focused on reproducibility.
- Utilizes the modular utilities from the `src` folder (dataset utils, training, and testing).
- **Setup:**
  1. Clone the repo.
  2. Add a `.env` file in the root directory with:
     ```
     HUGGINGFACE_TOKEN=your_token
     WANDB_API_KEY=your_key
     ```
  3. Run the notebook for an easy walkthrough!

---

## GitHub Actions

Automation is key, so I’ve set up workflows in `.github/workflows`:

### 1. **Test**

- Automatically triggered on every push.
- Validates the model’s performance against the development code.

### 2. **Push to HuggingFace**

- Manually triggered.
- Uploads new models from `src/model` to the HuggingFace repo.

### 3. **Train**

- Manually triggered.
- Requires a GPU-based environment (doesn’t work directly on GitHub Actions).

Check out the HuggingFace model repo: [AdaMata TrashNet on HuggingFace](https://huggingface.co/grediiiii/trashnet-adamata).

---

## Logs and Monitoring

For training logs and model performance:

- Visit [Wandb Logs](https://wandb.ai/mgradyn/trashnet-classification/runs/x74uwf2n?nw=nwusermgradyn)
  - Includes loss, accuracy, precision (macro & micro), recall (macro & micro), F1 (macro & micro), and confusion matrix history.

---

## Final Thoughts

I hope you enjoy exploring the code and learning from it as much as I enjoyed creating it. Feedback are always welcome!

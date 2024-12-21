import os
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, verbose=True, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves the model when validation loss improves."""
        if self.verbose:
            print(f"Validation loss improved. Saving model to model/{self.path}")

        model_dir = os.path.join(os.path.dirname(__file__), "model")
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, self.path)
        
        torch.save(model.state_dict(), model_path)
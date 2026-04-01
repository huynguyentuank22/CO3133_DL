"""
Unified trainer for RNN and Transformer models.
Supports training loop, early stopping, checkpointing, logging.
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from src import config
from src.metrics import compute_metrics
from src.utils import get_logger, save_checkpoint

logger = get_logger(__name__)


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then linear decay scheduler."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) /
                   float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Unified trainer for both RNN and Transformer models."""

    def __init__(self, model, optimizer, criterion, device=config.DEVICE,
                 model_type: str = "rnn", model_name: str = "model",
                 scheduler=None, grad_clip: float | None = None,
                 patience: int = config.EARLY_STOPPING_PATIENCE,
                 use_wandb: bool = False, wandb_run=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_type = model_type  # "rnn" or "transformer"
        self.model_name = model_name
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.patience = patience
        self.best_metric = -1.0
        self.epochs_no_improve = 0
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
                        "val_macro_f1": []}

    def _get_current_lr(self):
        if self.optimizer is None:
            return None
        if not self.optimizer.param_groups:
            return None
        return self.optimizer.param_groups[0].get("lr")

    def _train_epoch_rnn(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(dataloader.dataset)

    def _train_epoch_transformer(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item() * input_ids.size(0)
        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def _eval_rnn(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
        avg_loss = total_loss / len(dataloader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics, all_preds, all_labels

    @torch.no_grad()
    def _eval_transformer(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics, all_preds, all_labels

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, checkpoint_dir: str = config.CHECKPOINT_DIR) -> dict:
        """Full training loop with early stopping."""
        train_fn = (self._train_epoch_rnn if self.model_type == "rnn"
                    else self._train_epoch_transformer)
        eval_fn = (self._eval_rnn if self.model_type == "rnn"
                   else self._eval_transformer)

        best_ckpt_path = os.path.join(checkpoint_dir, f"{self.model_name}_best.pt")
        train_start = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = train_fn(train_loader)
            val_loss, val_metrics, _, _ = eval_fn(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            # Early stopping on macro_f1
            current_metric = val_metrics["macro_f1"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                save_checkpoint(self.model, self.optimizer, epoch,
                                val_metrics, best_ckpt_path)
                logger.info(f"  ✓ New best Macro-F1: {self.best_metric:.4f} — saved checkpoint")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info(f"  Early stopping triggered after {epoch} epochs")
                    if self.use_wandb and self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_accuracy": val_metrics["accuracy"],
                                "val_macro_f1": val_metrics["macro_f1"],
                                "best_val_macro_f1": self.best_metric,
                                "learning_rate": self._get_current_lr(),
                            },
                            step=epoch,
                        )
                    break

            if self.use_wandb and self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "best_val_macro_f1": self.best_metric,
                        "learning_rate": self._get_current_lr(),
                    },
                    step=epoch,
                )

        train_time = time.time() - train_start
        self.history["train_time_sec"] = train_time
        logger.info(f"Training complete in {train_time:.1f}s. Best Macro-F1: {self.best_metric:.4f}")

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.summary["best_val_macro_f1"] = self.best_metric
            self.wandb_run.summary["train_time_sec"] = train_time

        return self.history

    def evaluate(self, dataloader: DataLoader):
        """Evaluate on a dataset, return metrics, predictions, labels."""
        eval_fn = (self._eval_rnn if self.model_type == "rnn"
                   else self._eval_transformer)
        return eval_fn(dataloader)

    @torch.no_grad()
    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """Get softmax probabilities for all samples."""
        self.model.eval()
        all_probs = []
        if self.model_type == "rnn":
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        else:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

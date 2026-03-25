"""
Efficiency measurement: parameters, model size, train/inference time.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src import config
from src.utils import count_parameters, model_size_mb, get_logger

logger = get_logger(__name__)


def measure_inference_time(model, dataloader: DataLoader, model_type: str = "rnn",
                           device=config.DEVICE, num_warmup: int = 2) -> dict:
    """Measure inference time on a dataset."""
    model.eval()
    model.to(device)

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_warmup:
                break
            if model_type == "rnn":
                x = batch[0].to(device)
                _ = model(x)
            else:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                _ = model(ids, attention_mask=mask)

    # Actual measurement
    total_time = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "rnn":
                x = batch[0].to(device)
                bs = x.size(0)
                start = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end = time.perf_counter()
            else:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                bs = ids.size(0)
                start = time.perf_counter()
                _ = model(ids, attention_mask=mask)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end = time.perf_counter()
            total_time += (end - start)
            total_samples += bs

    avg_time_ms = (total_time / total_samples) * 1000

    return {
        "total_inference_time_sec": round(total_time, 4),
        "total_samples": total_samples,
        "inference_time_per_sample_ms": round(avg_time_ms, 4),
    }


def get_efficiency_stats(model, model_name: str, dataloader: DataLoader,
                         model_type: str = "rnn",
                         train_time_sec: float = 0.0) -> dict:
    """Get complete efficiency statistics for a model."""
    num_params = count_parameters(model)
    size_mb = model_size_mb(model)
    timing = measure_inference_time(model, dataloader, model_type)

    return {
        "model_name": model_name,
        "num_parameters": num_params,
        "model_size_mb": round(size_mb, 2),
        "train_time_sec": round(train_time_sec, 2),
        **timing,
    }


def create_efficiency_table(all_stats: list,
                             save_path: str | None = None) -> pd.DataFrame:
    """Create efficiency comparison table."""
    df = pd.DataFrame(all_stats)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Efficiency table saved to {save_path}")
    return df

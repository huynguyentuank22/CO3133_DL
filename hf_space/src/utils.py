"""
General utility functions: seeding, logging, checkpointing.
"""
import os
import random
import logging
import numpy as np
import torch
from src import config


def set_seed(seed: int = config.SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_checkpoint(model, optimizer, epoch: int, metrics: dict,
                    filepath: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, filepath)


def load_checkpoint(filepath: str, model, optimizer=None):
    """Load a training checkpoint."""
    ckpt = torch.load(filepath, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model) -> float:
    """Estimate model size in MB."""
    param_size = sum(p.nelement() * p.element_size()
                     for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size()
                      for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def load_dotenv_file(dotenv_path: str | None = None, override: bool = False) -> dict[str, str]:
    """Load key-value pairs from a .env file into os.environ."""
    if dotenv_path is None:
        dotenv_path = os.path.join(config.PROJECT_ROOT, ".env")

    loaded: dict[str, str] = {}
    if not os.path.exists(dotenv_path):
        return loaded

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                value = value[1:-1]

            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value

    return loaded


def resolve_wandb_config(enable_flag: bool,
                         project: str | None = None,
                         entity: str | None = None) -> dict[str, str | bool | None]:
    """Resolve runtime W&B settings from args + environment (.env supported)."""
    load_dotenv_file()

    api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB")
    if api_key:
        api_key = api_key.strip()
        os.environ["WANDB_API_KEY"] = api_key

    resolved_project = project or os.getenv("WANDB_PROJECT") or "co3133_dl"
    resolved_entity = entity if entity is not None else os.getenv("WANDB_ENTITY")
    enabled = bool(enable_flag or api_key)

    return {
        "enabled": enabled,
        "api_key": api_key,
        "project": resolved_project,
        "entity": resolved_entity,
    }

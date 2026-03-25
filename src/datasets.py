"""
PyTorch Dataset classes for RNN and Transformer models.
"""
from typing import List
import torch
from torch.utils.data import Dataset
import pandas as pd
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src import config


class RNNDataset(Dataset):
    """Dataset for BiLSTM / BiLSTM+Attention models."""

    def __init__(self, texts: List[str], labels: List[int],
                 vocab: Vocabulary, max_length: int = config.RNN_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_length)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class TransformerDataset(Dataset):
    """Dataset for DistilBERT / BERT models."""

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_length: int = config.TRANSFORMER_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_rnn_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, vocab: Vocabulary):
    """Create RNN datasets from dataframes."""
    train_ds = RNNDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), vocab)
    val_ds = RNNDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), vocab)
    test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
    return train_ds, val_ds, test_ds


def create_transformer_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                test_df: pd.DataFrame, model_name: str):
    """Create Transformer datasets from dataframes."""
    tokenizer = get_transformer_tokenizer(model_name)
    train_ds = TransformerDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds = TransformerDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
    return train_ds, val_ds, test_ds, tokenizer

"""
Transformer models: DistilBERT and BERT-base wrappers with fine-tuning strategies.
"""
import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, DistilBertConfig,
    BertModel, BertConfig,
)
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class DistilBertClassifier(nn.Module):
    """DistilBERT for sequence classification with fine-tune strategy support."""

    def __init__(self, model_name: str = config.DISTILBERT_MODEL_NAME,
                 num_classes: int = config.NUM_CLASSES,
                 finetune_strategy: str = config.FINETUNE_STRATEGY):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.finetune_strategy = finetune_strategy
        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        if self.finetune_strategy == "freeze":
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("DistilBERT: backbone frozen, training head only")
        elif self.finetune_strategy == "full":
            logger.info("DistilBERT: full fine-tuning")
        elif self.finetune_strategy == "llrd":
            logger.info("DistilBERT: LLRD strategy applied via optimizer")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]  # CLS token
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits

    def get_llrd_param_groups(self, base_lr: float = config.TRANSFORMER_LR,
                              decay_factor: float = config.LLRD_DECAY_FACTOR):
        """Return parameter groups with layer-wise learning rate decay."""
        param_groups = []
        # Classifier head
        param_groups.append({
            "params": list(self.classifier.parameters()) + list(self.dropout.parameters()),
            "lr": base_lr,
        })
        # Transformer layers (reverse order: last layer gets highest LR)
        num_layers = len(self.backbone.transformer.layer)
        for i in range(num_layers - 1, -1, -1):
            layer_lr = base_lr * (decay_factor ** (num_layers - 1 - i))
            param_groups.append({
                "params": list(self.backbone.transformer.layer[i].parameters()),
                "lr": layer_lr,
            })
        # Embeddings
        emb_lr = base_lr * (decay_factor ** num_layers)
        param_groups.append({
            "params": list(self.backbone.embeddings.parameters()),
            "lr": emb_lr,
        })
        return param_groups


class BertClassifier(nn.Module):
    """BERT-base for sequence classification with fine-tune strategy support."""

    def __init__(self, model_name: str = config.BERT_MODEL_NAME,
                 num_classes: int = config.NUM_CLASSES,
                 finetune_strategy: str = config.FINETUNE_STRATEGY):
        super().__init__()
        self.backbone = BertModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.finetune_strategy = finetune_strategy
        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        if self.finetune_strategy == "freeze":
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("BERT: backbone frozen, training head only")
        elif self.finetune_strategy == "full":
            logger.info("BERT: full fine-tuning")
        elif self.finetune_strategy == "llrd":
            logger.info("BERT: LLRD strategy applied via optimizer")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        pooled = outputs.pooler_output  # (batch, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def get_llrd_param_groups(self, base_lr: float = config.TRANSFORMER_LR,
                              decay_factor: float = config.LLRD_DECAY_FACTOR):
        """Return parameter groups with layer-wise learning rate decay."""
        param_groups = []
        # Classifier head
        param_groups.append({
            "params": list(self.classifier.parameters()) + list(self.dropout.parameters()),
            "lr": base_lr,
        })
        # Encoder layers
        num_layers = len(self.backbone.encoder.layer)
        for i in range(num_layers - 1, -1, -1):
            layer_lr = base_lr * (decay_factor ** (num_layers - 1 - i))
            param_groups.append({
                "params": list(self.backbone.encoder.layer[i].parameters()),
                "lr": layer_lr,
            })
        # Embeddings + pooler
        emb_lr = base_lr * (decay_factor ** num_layers)
        emb_params = list(self.backbone.embeddings.parameters())
        if hasattr(self.backbone, "pooler") and self.backbone.pooler is not None:
            emb_params += list(self.backbone.pooler.parameters())
        param_groups.append({
            "params": emb_params,
            "lr": emb_lr,
        })
        return param_groups

"""
model.py -- Multi-task transformer: one shared encoder with a single linear head per task.

Headline architecture used in the paper: a shared MARBERTv2 encoder produces a [CLS]
representation that feeds nine independent linear classification heads.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_classes_per_task: dict, dropout: float = 0.1):
        """
        Args:
            encoder_name: HuggingFace model name (e.g. "UBC-NLP/MARBERTv2").
            num_classes_per_task: dict {task_name: num_classes}.
            dropout: dropout applied to the pooled [CLS] representation.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.task_names = list(num_classes_per_task.keys())
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, n_cls)
            for task, n_cls in num_classes_per_task.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token
        return {task: self.heads[task](pooled) for task in self.task_names}

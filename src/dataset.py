"""
dataset.py — PyTorch Dataset for AAFAQ
Fixed after preprocessing. No changes needed between iterations.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


TASK_COLUMNS = [
    "Intent", "AnswerType", "CognitiveLevel", "QuestionParticleType",
    "QuestionType", "Subjectivity", "TemporalContext", "PurposeContext", "List"
]


class AAFAQDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_name: str, max_length: int = 128,
                 active_tasks: list = None):
        self.df = pd.read_csv(csv_path, encoding="utf-8")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.active_tasks = active_tasks or TASK_COLUMNS
        self.encoded_cols = [f"{t}_encoded" for t in self.active_tasks]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["QuestionText"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # Add all active task labels
        for task, col in zip(self.active_tasks, self.encoded_cols):
            item[task] = torch.tensor(int(row[col]), dtype=torch.long)

        return item


def get_num_classes(encoder_path: str, active_tasks: list = None) -> dict:
    import json
    with open(encoder_path) as f:
        encoders = json.load(f)
    tasks = active_tasks or TASK_COLUMNS
    return {t: encoders[t]["num_classes"] for t in tasks}

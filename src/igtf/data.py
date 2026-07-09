"""Dataset loading utilities for cached 9-d intent data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

DATASET_DATA_DIRS = {
    "weibo": "weibodata",
    "gossip": "gossipdata",
    "politifact": "politifactdata",
    "snopes": "snopesdata",
}


def dataset_data_dir(data_root: str | Path, dataset_name: str) -> Path:
    """Return the required ``<dataset>data`` directory."""
    if dataset_name not in DATASET_DATA_DIRS:
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {sorted(DATASET_DATA_DIRS)}")
    return Path(data_root) / DATASET_DATA_DIRS[dataset_name]


def load_intent_json(path: str | Path) -> List[Dict[str, Any]]:
    """Load dict/list JSON files with text, label, and 9-d intent_vector fields."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    rows: Iterable[Dict[str, Any]] = raw.values() if isinstance(raw, dict) else raw

    data: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        vector = item.get("intent_vector")
        text = item.get("text") or item.get("input_text") or item.get("claim_text") or ""
        if not isinstance(vector, list) or len(vector) != 9 or not text:
            continue
        data.append(
            {
                "text": str(text),
                "intent_vector": [float(x) for x in vector],
                "label": int(item["label"]),
                "text_hash": str(item.get("text_hash") or idx),
                "reasoning": str(item.get("reasoning", "")),
                "key_features": item.get("key_features", []),
            }
        )
    return data


def load_dataset_split(data_root: str | Path, dataset_name: str, split: str) -> List[Dict[str, Any]]:
    path = dataset_data_dir(data_root, dataset_name) / f"{split}.json"
    return load_intent_json(path)


class IntentTextDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "intent_vector": torch.tensor(row["intent_vector"], dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }

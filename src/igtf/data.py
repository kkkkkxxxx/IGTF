"""Data loading and intent-balanced splitting for cached 9-d data."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep cleaning/splitting usable without training deps.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from .data_cleaning import clean_intent_rows, dominant_intent, load_intent_json

SPLITS = ("train", "val", "test")

DATASET_DATA_DIRS = {
    "weibo": "weibodata",
    "weibodata": "weibodata",
    "gossip": "gossipdata",
    "gossipcop": "gossipdata",
    "gossipdata": "gossipdata",
    "politifact": "politifactdata",
    "politifactdata": "politifactdata",
    "snopes": "snopesdata",
    "snopesdata": "snopesdata",
}

MODEL_ALIASES = {
    "1": "1",
    "gpt": "1",
    "gpt55": "1",
    "gpt5.5": "1",
    "gpt-5.5": "1",
    "2": "2",
    "qwen": "2",
    "qwen2.5": "2",
    "qwen-2.5": "2",
    "3": "3",
    "llama": "3",
    "llama3": "3",
    "llama-3": "3",
}

AVAILABLE_MODELS = ("1", "2", "3")
AVAILABLE_DATASETS = ("weibo", "gossip", "politifact", "snopes")
AVAILABLE_DATASET_SPECS = tuple(f"{model}/{dataset}" for model in AVAILABLE_MODELS for dataset in AVAILABLE_DATASETS)


def dataset_data_dir(data_root: str | Path, dataset_name: str) -> Path:
    """Return the legacy ``<dataset>data`` directory when it exists."""
    dataset_dir = DATASET_DATA_DIRS.get(dataset_name.lower())
    if dataset_dir is None:
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {AVAILABLE_DATASETS}")
    return Path(data_root) / dataset_dir


def parse_dataset_spec(dataset_name: str, model_name: str = "1") -> Tuple[str, str, str]:
    """Normalize model/dataset input into ``(model_id, dataset_key, dataset_dir)``."""
    raw_dataset = dataset_name.strip().lower()
    raw_model = model_name.strip().lower()
    if "/" in raw_dataset:
        raw_model, raw_dataset = raw_dataset.split("/", 1)
    elif ":" in raw_dataset:
        raw_model, raw_dataset = raw_dataset.split(":", 1)

    model_id = MODEL_ALIASES.get(raw_model)
    if model_id is None:
        raise ValueError(f"Unknown model {model_name!r}; choose from {AVAILABLE_MODELS}")

    dataset_dir = DATASET_DATA_DIRS.get(raw_dataset)
    if dataset_dir is None:
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {AVAILABLE_DATASETS}")

    dataset_key = dataset_dir.removesuffix("data")
    return model_id, dataset_key, dataset_dir


def resolve_merged_data_file(data_root: str | Path, dataset_name: str, model_name: str = "1") -> Path:
    """Resolve model/dataset aliases to ``intent_data/<model>/<dataset>data/all.json``."""
    root = Path(data_root)
    model_id, _, dataset_dir = parse_dataset_spec(dataset_name, model_name)
    path = root / model_id / dataset_dir / "all.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing merged intent file: {path}")
    return path


def load_model_dataset(data_root: str | Path, dataset_name: str, model_name: str = "1") -> List[Dict[str, Any]]:
    return load_intent_json(resolve_merged_data_file(data_root, dataset_name, model_name))


def _split_counts(n_items: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if n_items <= 0:
        return (0, 0, 0)
    raw = [n_items * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = n_items - sum(counts)
    order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if n_items >= len(active):
        for i in active:
            if counts[i] == 0:
                donor = max(active, key=lambda j: counts[j])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[i] += 1
    return (counts[0], counts[1], counts[2])


def split_intent_balanced(
    rows: List[Dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split cleaned rows by label and dominant intent for balanced intent coverage."""
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    ratios = (train_ratio / total, val_ratio / total, test_ratio / total)

    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["label"]), dominant_intent(row))].append(row)

    rng = random.Random(seed)
    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLITS}
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        n_train, n_val, n_test = _split_counts(len(items), ratios)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val : n_train + n_val + n_test])

    for offset, split in enumerate(SPLITS):
        random.Random(seed + 100 + offset).shuffle(splits[split])
    return splits


@lru_cache(maxsize=32)
def _cached_split(
    data_root: str,
    model_name: str,
    dataset_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    rows = load_model_dataset(data_root, dataset_name, model_name)
    splits = split_intent_balanced(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return tuple(splits["train"]), tuple(splits["val"]), tuple(splits["test"])


def load_dataset_split(
    data_root: str | Path,
    dataset_name: str,
    split: str,
    *,
    model_name: str = "1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load a split from legacy split files or from merged model data."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")

    if "/" not in dataset_name and ":" not in dataset_name:
        dataset_dir = DATASET_DATA_DIRS.get(dataset_name.lower())
        legacy_path = Path(data_root) / str(dataset_dir) / f"{split}.json"
        if legacy_path.exists():
            return load_intent_json(legacy_path)

    train_rows, val_rows, test_rows = _cached_split(
        str(Path(data_root)),
        model_name,
        dataset_name,
        float(train_ratio),
        float(val_ratio),
        float(test_ratio),
        int(seed),
    )
    return {"train": list(train_rows), "val": list(val_rows), "test": list(test_rows)}[split]


def write_clean_splits(
    data_root: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    *,
    model_name: str = "1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """Materialize cleaned, intent-balanced train/val/test JSON files."""
    rows = load_model_dataset(data_root, dataset_name, model_name)
    splits = split_intent_balanced(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with open(output_root / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(split_rows, f, ensure_ascii=False, indent=2)
    return {split: len(split_rows) for split, split_rows in splits.items()}


class IntentTextDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("IntentTextDataset requires torch; install training dependencies first.")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean merged 9-d intent data and create balanced splits.")
    parser.add_argument("--data-root", default="intent_data")
    parser.add_argument("--model", default="1", help="Model id or alias, e.g. 1/gpt55, 2/qwen, 3/llama.")
    parser.add_argument("--dataset", required=True, help="Dataset name, or a combined spec such as 1/weibo.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = write_clean_splits(
        args.data_root,
        args.dataset,
        args.output_dir,
        model_name=args.model,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

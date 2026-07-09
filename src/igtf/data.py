"""Data cleaning, loading, and intent-balanced splitting for cached 9-d data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep cleaning/splitting usable without training deps.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

INTENT_DIM = 9
SPLITS = ("train", "val", "test")

DATASET_DATA_DIRS = {
    "weibo": "weibodata",
    "gossip": "gossipdata",
    "politifact": "politifactdata",
    "snopes": "snopesdata",
}

MODEL_DATA_FILES = {
    "1": "1/all.json",
    "gpt": "1/all.json",
    "gpt55": "1/all.json",
    "gpt5.5": "1/all.json",
    "weibo": "1/all.json",
    "2": "2/all.json",
    "qwen": "2/all.json",
    "qwen2.5": "2/all.json",
    "politifact": "2/all.json",
    "3": "3/all.json",
    "llama": "3/all.json",
    "llama3": "3/all.json",
    "gossip": "3/all.json",
}

AVAILABLE_DATASETS = sorted(MODEL_DATA_FILES)

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "false": 0,
    "fake": 0,
    "pants-fire": 0,
    "barely-true": 0,
    "mostly-false": 0,
    "true": 1,
    "real": 1,
    "mostly-true": 1,
    "half-true": 1,
}


def dataset_data_dir(data_root: str | Path, dataset_name: str) -> Path:
    """Return the legacy ``<dataset>data`` directory when it exists."""
    if dataset_name not in DATASET_DATA_DIRS:
        raise ValueError(f"Unknown legacy dataset {dataset_name!r}; choose from {sorted(DATASET_DATA_DIRS)}")
    return Path(data_root) / DATASET_DATA_DIRS[dataset_name]


def _iter_json_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, Mapping):
        rows = raw.values()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        rows = raw
    else:
        rows = []
    for item in rows:
        if isinstance(item, dict):
            yield dict(item)


def _normalize_text(item: Mapping[str, Any]) -> str:
    text = (
        item.get("text")
        or item.get("input_text")
        or item.get("claim_text")
        or item.get("claim")
        or item.get("content")
        or ""
    )
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _normalize_label(item: Mapping[str, Any]) -> int | None:
    raw = item.get("label")
    if raw is None:
        raw = item.get("binary_label", item.get("cred_label", item.get("veracity")))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) in (0, 1):
            return int(raw)
        return None
    key = str(raw).strip().lower()
    return LABEL_MAP.get(key)


def _normalize_intent_vector(item: Mapping[str, Any]) -> List[float] | None:
    vector = item.get("intent_vector")
    if not isinstance(vector, list) or len(vector) != INTENT_DIM:
        return None
    cleaned: List[float] = []
    for value in vector:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        cleaned.append(min(1.0, max(0.0, number)))
    return cleaned


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dominant_intent(row: Mapping[str, Any]) -> int:
    vector = row["intent_vector"]
    return max(range(INTENT_DIM), key=lambda idx: float(vector[idx]))


def clean_intent_rows(rows: Iterable[Mapping[str, Any]], *, deduplicate: bool = True) -> List[Dict[str, Any]]:
    """Clean records and keep only valid text, binary label, and 9-d intent rows."""
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(rows):
        text = _normalize_text(item)
        label = _normalize_label(item)
        vector = _normalize_intent_vector(item)
        if not text or label is None or vector is None:
            continue

        text_hash = str(item.get("text_hash") or _stable_hash(text))
        dedupe_key = text_hash if text_hash else _stable_hash(text)
        if deduplicate and dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        row = dict(item)
        row.update(
            {
                "text": text,
                "intent_vector": vector,
                "label": int(label),
                "text_hash": text_hash or str(idx),
            }
        )
        cleaned.append(row)
    return cleaned


def load_intent_json(path: str | Path, *, deduplicate: bool = True) -> List[Dict[str, Any]]:
    """Load and clean dict/list JSON files containing cached 9-d intent vectors."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return clean_intent_rows(_iter_json_records(raw), deduplicate=deduplicate)


def resolve_merged_data_file(data_root: str | Path, dataset_name: str) -> Path:
    """Resolve model-id aliases such as ``1``, ``qwen``, or ``llama`` to ``all.json``."""
    root = Path(data_root)
    key = dataset_name.lower()
    direct_dir = root / dataset_name / "all.json"
    direct_file = root / f"{dataset_name}.json"
    if direct_dir.exists():
        return direct_dir
    if direct_file.exists():
        return direct_file
    if key not in MODEL_DATA_FILES:
        raise ValueError(f"Unknown dataset/model {dataset_name!r}; choose from {AVAILABLE_DATASETS}")
    path = root / MODEL_DATA_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing merged intent file: {path}")
    return path


def load_model_dataset(data_root: str | Path, dataset_name: str) -> List[Dict[str, Any]]:
    return load_intent_json(resolve_merged_data_file(data_root, dataset_name))


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
    dataset_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    rows = load_model_dataset(data_root, dataset_name)
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
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load a split from legacy split files or from merged model data."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")

    if dataset_name in DATASET_DATA_DIRS:
        legacy_path = Path(data_root) / DATASET_DATA_DIRS[dataset_name] / f"{split}.json"
        if legacy_path.exists():
            return load_intent_json(legacy_path)

    train_rows, val_rows, test_rows = _cached_split(
        str(Path(data_root)),
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
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """Materialize cleaned, intent-balanced train/val/test JSON files."""
    rows = load_model_dataset(data_root, dataset_name)
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
    parser.add_argument("--dataset", required=True, help="Model id or alias, e.g. 1, 2, 3, gpt55, qwen, llama.")
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

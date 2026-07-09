"""Cleaning and validation helpers for cached 9-d intent records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

INTENT_DIM = 9

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


def iter_json_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, Mapping):
        rows = raw.values()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        rows = raw
    else:
        rows = []
    for item in rows:
        if isinstance(item, dict):
            yield dict(item)


def normalize_text(item: Mapping[str, Any]) -> str:
    text = (
        item.get("text")
        or item.get("input_text")
        or item.get("claim_text")
        or item.get("claim")
        or item.get("content")
        or ""
    )
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_label(item: Mapping[str, Any]) -> int | None:
    raw = item.get("label")
    if raw is None:
        raw = item.get("binary_label", item.get("cred_label", item.get("veracity")))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) in (0, 1):
            return int(raw)
        return None
    return LABEL_MAP.get(str(raw).strip().lower())


def normalize_intent_vector(item: Mapping[str, Any]) -> List[float] | None:
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


def stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dominant_intent(row: Mapping[str, Any]) -> int:
    vector = row["intent_vector"]
    return max(range(INTENT_DIM), key=lambda idx: float(vector[idx]))


def clean_intent_rows(rows: Iterable[Mapping[str, Any]], *, deduplicate: bool = True) -> List[Dict[str, Any]]:
    """Keep only rows with valid text, binary label, and normalized 9-d vectors."""
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, item in enumerate(rows):
        text = normalize_text(item)
        label = normalize_label(item)
        vector = normalize_intent_vector(item)
        if not text or label is None or vector is None:
            continue

        text_hash = str(item.get("text_hash") or stable_text_hash(text))
        dedupe_key = text_hash if text_hash else stable_text_hash(text)
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
    return clean_intent_rows(iter_json_records(raw), deduplicate=deduplicate)

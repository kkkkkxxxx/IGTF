#!/usr/bin/env python
"""Train IGTF on one cached 9-d intent dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from igtf.data import IntentTextDataset, load_dataset_split
from igtf.model import IGTFClassifier, IGTFConfig


def default_bert_model(dataset_name: str) -> str:
    if dataset_name == "weibo":
        return "bert-base-chinese"
    return "bert-base-uncased"


@torch.no_grad()
def evaluate(model: IGTFClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    predictions = []
    gold_labels = []
    total_loss = 0.0
    for batch in loader:
        batch = normalize_batch(batch)
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        total_loss += float(output["loss"].item())
        predictions.extend(output["predictions"].cpu().tolist())
        gold_labels.extend(batch["labels"].cpu().tolist())
    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(gold_labels, predictions),
        "macro_f1": f1_score(gold_labels, predictions, average="macro"),
    }


def normalize_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "intent_vector": batch["intent_vector"],
        "labels": batch["label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IGTF with cached 9-d intent data.")
    parser.add_argument("--dataset", choices=["weibo", "gossip", "politifact", "snopes"], required=True)
    parser.add_argument("--data-root", default="intent_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-intent-game", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = args.bert_model or default_bert_model(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    train_rows = load_dataset_split(args.data_root, args.dataset, "train")
    val_rows = load_dataset_split(args.data_root, args.dataset, "val")
    test_rows = load_dataset_split(args.data_root, args.dataset, "test")
    print(f"Loaded {args.dataset}: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_loader = DataLoader(
        IntentTextDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IntentTextDataset(val_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        IntentTextDataset(test_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    model = IGTFClassifier(
        IGTFConfig(
            bert_model=bert_model,
            use_intent_game=not args.no_intent_game,
        )
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_root = Path(args.output_dir) / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_path = output_root / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = normalize_batch(batch)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    with open(output_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_val, "test": test_metrics}, f, indent=2)
    print(json.dumps({"best_val_macro_f1": best_val, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()

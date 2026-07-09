# IGTF

This repository contains a clean release of the code and cached 9-dimensional
intent data for **IGTF: Game-Inspired Structured Intent Calibration for
LLM-Assisted Fake News Detection**.

## Contents

- `src/igtf/`: model code for text MoE encoding, 9-d intent game refinement,
  intent-text fusion, and classification.
- `scripts/train_igtf.py`: a compact training entry point for cached intent
  data.
- `intent_data/1/all.json`: merged GPT-5.5/default 9-d intent data.
- `intent_data/2/all.json`: merged Qwen 9-d intent data.
- `intent_data/3/all.json`: merged LLaMA 9-d intent data.

Each model directory stores one merged file. `src/igtf/data.py` cleans the
merged records and creates train/val/test splits with balanced label and
dominant-intent distributions at load time.

## Data Format

Each JSON item contains:

```json
{
  "text": "...",
  "intent_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "label": 0,
  "text_hash": "...",
  "reasoning": "...",
  "key_features": []
}
```

Intent order:

1. `public-oriented`
2. `emotion-driven`
3. `individual-focused`
4. `popularize`
5. `clout-seeking`
6. `conflict-creation`
7. `smearing`
8. `bias-injection`
9. `connection-seeking`

## Quick Start

```bash
pip install -r requirements.txt
set PYTHONPATH=%CD%\src
python scripts/train_igtf.py --dataset 3 --epochs 5 --batch-size 16
```

For Linux/macOS:

```bash
export PYTHONPATH=$PWD/src
python scripts/train_igtf.py --dataset 3 --epochs 5 --batch-size 16
```

Available model ids are `1`, `2`, and `3`; aliases such as `gpt55`, `qwen`,
and `llama` are also supported.

## Notes

The released training code uses cached local intent vectors only. It does not
call remote LLM services and does not store private credentials.

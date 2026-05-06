# Alignment via Direct Preference Optimization (DPO)

CENG 467 — Natural Language Understanding and Generation  
İzmir Institute of Technology, Spring 2026

## Overview

This project implements DPO (Direct Preference Optimization) for aligning a language model with human preferences, using the Anthropic HH-RLHF dataset.

**Models compared:**
- Zero-shot pretrained baseline (TinyLlama-1.1B)
- SFT (Supervised Fine-Tuning) baseline
- DPO fine-tuned model

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Results

### 1. Prepare Dataset
```bash
python data/prepare_dataset.py
# For fast dev iteration (5000 samples):
python data/prepare_dataset.py --subset 5000
```

### 2. Train SFT Baseline
```bash
python scripts/train_sft.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 3. Train DPO Model
```bash
python scripts/train_dpo.py --ref_model_path ./models/checkpoints/sft --beta 0.1
```

### 4. Evaluate All Models
```bash
python scripts/evaluate.py
# Results saved to results/tables/eval_results.json
```

## Project Structure

```
├── data/               # Dataset preparation scripts
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Exploratory notebooks
├── models/checkpoints/ # Saved model weights (gitignored)
├── results/tables/     # Metric results (CSV/JSON)
├── report/             # LaTeX report
└── requirements.txt
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | TinyLlama-1.1B | Fits T4 GPU with LoRA |
| LoRA rank | 16 | Ablation: [8, 16, 32] |
| DPO β | 0.1 | Ablation: [0.01, 0.05, 0.1, 0.2, 0.5] |
| Learning rate | 5e-5 | |
| Batch size | 4 (SFT), 2 (DPO) | With grad accum × 4 |

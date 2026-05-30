# Alignment via Direct Preference Optimization (DPO)

CENG 467 — Natural Language Understanding and Generation  
İzmir Institute of Technology, Spring 2026

## Overview

This project implements DPO (Direct Preference Optimization) for aligning a language model with human preferences, using the Anthropic HH-RLHF dataset.

**Models compared:**
- Zero-shot pretrained baseline (TinyLlama-1.1B-Chat-v1.0)
- SFT (Supervised Fine-Tuning) baseline — trained on chosen responses
- DPO fine-tuned model — trained with contrastive preference loss

**Results (200 test samples):**

| Model | Perplexity ↓ | BERTScore F1 ↑ |
|-------|-------------|----------------|
| Zero-shot | 17.006 | 0.8294 |
| SFT (50k, 2 ep.) | 9.536 | 0.8171 |
| DPO β=0.1 (50k, 2 ep.) | 16.949 | 0.8262 |

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Results

All experiments were run on Google Colab Pro (A100 GPU). See `notebooks/` for interactive versions.

### 1. Train SFT Baseline

```bash
python scripts/train_sft.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir ./models/checkpoints/sft_50k \
  --subset 50000 \
  --epochs 2 \
  --lr 1e-4 \
  --lora_rank 16
```

### 2. Train DPO Model

```bash
python scripts/train_dpo.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ref_model_path ./models/checkpoints/sft_50k \
  --output_dir ./models/checkpoints/dpo_50k \
  --subset 50000 \
  --epochs 2 \
  --beta 0.1 \
  --lora_rank 16 \
  --lr 1e-5
```

### 3. Evaluate All Models

```bash
python scripts/run_eval.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sft_path ./models/checkpoints/sft_50k \
  --dpo_path ./models/checkpoints/dpo_50k \
  --n_samples 200
```

Results are saved to `results/tables/eval_results.json`.

### 4. Ablation Study

```bash
# Beta ablation
python scripts/train_dpo.py --beta 0.05 --subset 10000 --epochs 1 \
  --ref_model_path ./models/checkpoints/sft_50k \
  --output_dir ./models/checkpoints/dpo_beta005

# LoRA rank ablation
python scripts/train_dpo.py --lora_rank 8 --subset 10000 --epochs 1 \
  --ref_model_path ./models/checkpoints/sft_50k \
  --output_dir ./models/checkpoints/dpo_rank8

# Learning rate ablation
python scripts/train_dpo.py --lr 5e-5 --subset 10000 --epochs 1 \
  --ref_model_path ./models/checkpoints/sft_50k \
  --output_dir ./models/checkpoints/dpo_lr5e5
```

## Project Structure

```
├── data/                    # Dataset preparation scripts
│   └── prepare_dataset.py
├── scripts/                 # Training and evaluation scripts
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── run_eval.py
├── notebooks/               # Colab training notebooks
│   ├── sft_dpo_training_50k.ipynb
│   └── ablation_runs.ipynb
├── results/tables/          # Evaluation results (JSON)
├── report/                  # LaTeX report (LNCS format)
│   └── final_report.tex
└── requirements.txt
```

## Key Hyperparameters

| Parameter | SFT | DPO |
|-----------|-----|-----|
| Base model | TinyLlama-1.1B-Chat-v1.0 | TinyLlama-1.1B-Chat-v1.0 |
| Training samples | 50,000 | 50,000 |
| Epochs | 2 | 2 |
| Learning rate | 1e-4 | 1e-5 |
| LoRA rank | 16 | 16 |
| Batch size | 4 | 2 |
| β | — | 0.1 |

## Team

- Enes Ergün Hoşgör (300201053)
- Samet Buldanlıoğlu (300201085)
- Uğur Mert Çavuşoğlu (300201087)

İzmir Institute of Technology, Spring 2026

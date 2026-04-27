# CLAUDE.md — Alignment via Direct Preference Optimization (DPO)

## Project Identity

| Field | Value |
|-------|-------|
| Course | CENG 467 — Natural Language Understanding and Generation |
| Institution | İzmir Institute of Technology (IYTE) |
| Instructor | Prof. Dr. Aytuğ ONAN |
| Semester | Spring 2026 |
| Topic | #2 — Alignment via Direct Preference Optimization (DPO) |
| Group Size | 3 members |
| Dataset | `Anthropic/hh-rlhf` (HuggingFace Hub) |
| Primary Language | Python |

---

## Intelligent Behavior Rules

### Model Selection
Always select the model based on task complexity:

| Task Type | Model | Examples |
|-----------|-------|---------|
| Quick fixes | `claude-haiku-4-5` | typo, rename variable, single-line edit |
| Default work | `claude-sonnet-4-6` | code edits, debugging, metric scripts, file I/O |
| Deep reasoning | `claude-opus-4-7` | architecture decisions, paper analysis, complex research, DPO theory |

### When to Enter Plan Mode
**Always call `EnterPlanMode` before starting if ANY of these are true:**
- Task has 3 or more distinct steps
- Task touches more than 2 files
- Task involves architectural decisions (model choice, training loop design)
- Task is ambiguous or has multiple valid approaches
- Task involves the full training pipeline

**Skip plan mode for:** single-file edits, bug fixes, adding one function, metric computation.

### Response Style
- Be concise. One sentence updates, not paragraphs.
- When referencing code, use `file_path:line_number` format.
- Default to no comments in code unless the WHY is non-obvious.
- Never add features beyond what is asked.

---

## Technical Stack

```
transformers       # HuggingFace model loading, tokenizers
datasets           # HuggingFace dataset loading (hh-rlhf)
trl                # DPOTrainer, SFTTrainer
peft               # LoRA / QLoRA for memory-efficient fine-tuning
torch              # PyTorch backend
accelerate         # Multi-GPU / mixed precision training
evaluate           # HuggingFace evaluation metrics
bert_score         # BERTScore computation
nltk               # BLEU, tokenization utilities
scikit-learn       # Utility metrics
wandb              # Experiment tracking (optional but recommended)
```

### Recommended Base Model
- **TinyLlama-1.1B** — fits T4 (15GB) with LoRA, fast iteration
- Fallback: **GPT-2 medium** — even lighter, good for prototyping
- Upgrade path: **Llama-3.2-1B** if A100 available

### Environment
- Training: **Kaggle** (free, 30h/week P100) or **Colab Pro** (A100)
- Local: only for data prep, evaluation scripts, report writing
- Python 3.10+, CUDA 11.8+

---

## Project Pipeline

Work through these steps in order. Each step builds on the previous.

### Step 0 — Environment Setup
```bash
pip install transformers datasets trl peft accelerate evaluate bert_score torch wandb
```
Create `requirements.txt` immediately after setup.

### Step 1 — Dataset Loading & Preprocessing
```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")
# Raw format: {"chosen": "\n\nHuman: ...\n\nAssistant: ...", "rejected": "..."}
# DPOTrainer expects: {"prompt": str, "chosen": str, "rejected": str}

def extract_prompt_and_response(sample):
    # The chosen string contains the full conversation; split at last "\n\nAssistant:"
    prompt = sample["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    chosen_response = sample["chosen"].rsplit("\n\nAssistant:", 1)[1].strip()
    rejected_response = sample["rejected"].rsplit("\n\nAssistant:", 1)[1].strip()
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}

dataset = dataset.map(extract_prompt_and_response)
# Filter malformed samples
dataset = dataset.filter(lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
dataset = dataset.filter(lambda x: x["chosen"] != x["rejected"])
```
- Use a small subset first (e.g., 5000 samples) for fast iteration
- Full dataset: ~160k train samples — use all for final training

### Step 2 — Baseline 1: Zero-Shot Pretrained Model
- Load base model with NO fine-tuning
- Run inference on test prompts
- Compute: **perplexity**, **BERTScore**, qualitative samples
- This establishes the "unaligned" lower bound

### Step 3 — Baseline 2: SFT Model
```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                        # LoRA rank — ablation candidate
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="chosen",   # only learn from preferred responses
    args=SFTConfig(output_dir="./sft_checkpoint", num_train_epochs=2, fp16=True),
)
sft_trainer.train()
sft_trainer.save_model("./sft_checkpoint")  # THIS becomes ref_model for DPO
```
- Save the SFT checkpoint — it is used as `ref_model` in Step 4
- Same eval metrics as Baseline 1

### Step 4 — DPO Fine-Tuning (Main Contribution)
```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    beta=0.1,          # KL penalty — KEY hyperparameter for ablation
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=True,
)

trainer = DPOTrainer(
    model=model,            # fresh copy of base model (will be DPO-trained)
    ref_model=ref_model,    # load from "./sft_checkpoint" — kept frozen
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./dpo_checkpoint")
```
- Load `ref_model` from `./sft_checkpoint` (Step 3 output) — do NOT train it
- Apply same LoRA config as SFT step to the trainable model
- Save checkpoints after each epoch

### Step 5 — Evaluation
Run all three models (Zero-shot, SFT, DPO) through the same eval pipeline:

| Metric | Tool | What it measures |
|--------|------|-----------------|
| Perplexity | `evaluate` | Language model fluency |
| BERTScore (F1) | `bert_score` | Semantic similarity to reference |
| Win-rate | reward model scoring | DPO vs SFT preference |
| Qualitative | manual inspection | Coherence, safety, helpfulness |

### Step 6 — Ablation Study (10 pts — do not skip)
Vary one parameter at a time, keep others fixed:
- **β values**: `[0.01, 0.05, 0.1, 0.2, 0.5]` — most important
- **Learning rate**: `[1e-5, 5e-5, 1e-4]`
- **LoRA rank**: `[8, 16, 32]`
- Report results in a table

### Step 7 — Error Analysis
- Pick 20-30 failure cases from the test set
- Categorize: hallucination / refusal failure / off-topic / repetition
- Write 1-2 paragraphs for the paper's Error Analysis section

### Step 8 — Ethical Considerations
Required section in the report (5 pts):
- Bias in HH-RLHF labels (annotator demographics)
- Risk of over-refusal after alignment
- Potential for reward hacking in DPO
- Misuse: aligned model can still be harmful with adversarial prompts

---

## Grading Rubric — Priority Order

Focus effort proportionally to points:

| Priority | Criterion | Points |
|----------|-----------|--------|
| 1 | Model/Method Implementation (DPO correct) | **20** |
| 2 | Baseline Comparison (zero-shot + SFT) | **15** |
| 3 | Evaluation Methodology | **15** |
| 4 | Problem Formulation | 10 |
| 5 | Ablation Study | 10 |
| 6 | Generation Quality Analysis | 10 |
| 7 | Presentation & Demo | 10 |
| 8 | Ethical Considerations | 5 |
| 9 | Reproducibility & Code Quality | 5 |
| + | Bonus (extra model / paper reproduction) | +5 |

**Total: 100 + 5 bonus**

---

## Timeline & Milestones

| Week | Milestone | Status |
|------|-----------|--------|
| Week 7 | Group formation + topic selection + proposal | Done |
| Week 8-10 | Literature review + dataset prep + SFT baseline | In Progress |
| Week 11 | **CHECKPOINT**: working baselines + initial DPO results | Deadline |
| Week 12-14 | Ablation study + error analysis + paper writing | — |
| Week 15 | **FINAL**: LNCS paper (6-8 pages) + GitHub repo | Deadline |
| Week 15-16 | Presentation + live demo | — |

GitHub commits must be consistent — lack of activity negatively affects grade.

---

## Academic Report Structure

Follow this exactly (LNCS format, 6-8 pages):

1. **Introduction** — problem statement, why alignment matters, paper contributions
2. **Related Work** — RLHF, PPO vs DPO, Constitutional AI, InstructGPT
3. **Methodology** — DPO loss derivation, SFT baseline, LoRA setup
4. **Experimental Setup** — dataset stats, model specs, hyperparameters, hardware
5. **Results** — tables comparing zero-shot / SFT / DPO on all metrics
6. **Error Analysis** — failure cases, categorized
7. **Discussion** — what worked, what didn't, β sensitivity, limitations
8. **Conclusion** — summary + future work

Mandatory citations: Rafailov et al. 2023 (DPO paper), Anthropic HH-RLHF paper, TRL library.

---

## Reproducibility Checklist

Before final submission, verify:
- [ ] `requirements.txt` with pinned versions
- [ ] `README.md` with step-by-step reproduction instructions
- [ ] Dataset download script or instructions
- [ ] Training script with all hyperparameters as CLI args
- [ ] Evaluation script that outputs all metrics
- [ ] Saved model checkpoints (or HuggingFace Hub link)
- [ ] All random seeds set and documented
- [ ] GitHub repo is public with clean commit history

---

## DPO Theory Reference

Claude should be able to explain these concepts when asked:

**DPO Loss:**
```
L_DPO(π_θ) = -E[(x,y_w,y_l)] [ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

- `y_w` = chosen (winning) response
- `y_l` = rejected (losing) response
- `π_ref` = frozen reference model (SFT model)
- `β` = temperature controlling deviation from reference policy
- No explicit reward model needed — reward is implicit in preference data

**Why DPO over RLHF/PPO:**
- No separate reward model training
- No RL instability (no PPO rollouts)
- Same or better alignment quality
- Much simpler to implement and tune

**Key insight:** DPO reformulates the RLHF objective as a classification problem over preference pairs, solvable with standard supervised learning.

---

## Project File Structure (Target)

```
Alignment_via_DPO/
├── CLAUDE.md                    # This file
├── README.md                    # Reproduction instructions
├── requirements.txt
├── data/
│   └── prepare_dataset.py       # HH-RLHF download + preprocessing
├── models/
│   └── checkpoints/             # Saved model weights
├── scripts/
│   ├── train_sft.py             # Baseline 2: SFT training
│   ├── train_dpo.py             # Main: DPO training
│   └── evaluate.py              # All metrics computation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_dpo_training.ipynb
├── results/
│   └── tables/                  # CSV files with metric results
├── report/
│   └── ceng467_dpo_report.tex   # LNCS LaTeX report
└── ceng467_termProject.pdf      # Original assignment
```

---

## Git Branching Strategy

### Branch Structure
```
main          ← stable, only merged PRs land here
├── dev       ← integration branch, merge feature branches here first
├── feature/dataset-prep
├── feature/sft-baseline
├── feature/dpo-training
├── feature/evaluation
└── feature/ablation
```

### Rules
- **Never commit directly to `main`**
- All work goes on a `feature/` branch
- Open a PR → merge into `dev` first → when milestone is done, `dev` → `main`
- One person = one branch at a time (avoids merge conflicts)

### Suggested Assignment
| Branch | Owner |
|--------|-------|
| `feature/dataset-prep` + `feature/sft-baseline` | Person 1 |
| `feature/dpo-training` + `feature/ablation` | Person 2 |
| `feature/evaluation` + `feature/report` | Person 3 |

### Commands for Daily Work
```bash
# Start new work
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# Save progress
git add .
git commit -m "descriptive message"
git push origin feature/your-feature-name

# Then open PR on GitHub: feature/xxx → dev
```

---

## Dos and Don'ts

### Do
- Always run baselines before DPO — you need comparison numbers
- Save checkpoints frequently — Colab/Kaggle sessions die
- Use `wandb` or at minimum print metrics to a log file
- Commit to GitHub after every significant working change
- Keep β ablation results in a CSV — you'll need them for the paper

### Don't
- Don't try to train without LoRA on T4 — you will OOM
- Don't skip the ablation study — it's 10 points
- Don't use a model larger than 1.1B without A100
- Don't leave the report to the last week — write methodology as you implement
- Don't hardcode hyperparameters — use argparse or config files

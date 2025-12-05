# Membership Inference Attack on GPT-2

Membership inference attacks to determine if text samples were in a language model's training data.

---

## Overview

**Target Model:** GPT-2 fine-tuned on WikiText-103 + train_finetune.json
**Test Set:** 2,000 samples (1,000 members, 1,000 non-members)
**Goal:** Predict which samples were in training data

---

## Files

### Main Notebooks

| File | Purpose | Result |
|------|---------|--------|
| `MIA_LiRA.ipynb` | LiRA Gaussian with 32 shadows | ✅ 64.0% TPR |
| `MIA_zscore 32.ipynb` | Z-Score attack with 32 shadows | ✅ 73.0% TPR |
| `MIA_zscore.ipynb` | Z-Score attack with 62 shadows | ✅ **77.9% TPR** (best) |
| `prepare_shadow_models.ipynb` | Generate shadow models for Z-Score | ✅ Complete |
| `prepare_and_train_shadow_lira.ipynb` | Generate 32 shadow models for LiRA | ✅ Complete |

### Data Files

```
data/
├── train/
│   ├── train_finetune.json      # Target's fine-tuning data (30k samples)
│   ├── test.json                # Test set (2k samples)
│   └── test_label.json          # Ground truth (1=member, 0=non-member)
│
└── shadow_datasets_lira/
    ├── keep_matrix.npy          # Binary matrix [2000, 32]: which shadows saw which samples
    ├── test.json                # Copy of test set for shadows
    └── shadow_{0..31}/          # 32 shadow training datasets
```

### Model Files

```
models/
├── train/
│   └── gpt2_3_lora32_adamw_b8_lr2/    # Target model (LoRA weights)
│
└── shadow_lira/
    └── shadow_{0..31}/                 # 32 shadow models (LoRA weights)
```

---

## Attack Algorithms

### 1. Z-Score Attack

**How it works:**
```python
# Train shadow models that NEVER see test samples
# Can use 32 shadows or 62 shadows

# For each test sample:
for sample in test_set:
    target_loss = target_model.loss(sample)
    shadow_losses = [shadow.loss(sample) for shadow in shadows]  # All shadows

    # Compute z-score
    shadow_mean = mean(shadow_losses)
    shadow_std = std(shadow_losses)
    z_score = (shadow_mean - target_loss) / shadow_std

    # Higher z_score → target has lower loss → likely MEMBER
```

**Intuition:** If target saw the sample during training, its loss will be lower than shadows that never saw it.

**Results:**
- 32 shadows: **73.0%** TPR @ 1% FPR
- 62 shadows: **77.9%** TPR @ 1% FPR

**Run:**
```bash
# Z-Score with 32 shadows
jupyter notebook "MIA_zscore 32.ipynb"

# Z-Score with 62 shadows
jupyter notebook "MIA_zscore.ipynb"
```

---

### 2. LiRA (Gaussian Likelihood Ratio)

**How it works:**
```python
# Train 32 shadow models with per-example membership
# keep_matrix[j, s] = 1 if shadow s trained on sample j

for sample_j in test_set:
    # Get shadow losses for this specific sample
    in_shadows = [s where keep_matrix[j,s] == 1]   # 16 shadows that SAW sample j
    out_shadows = [s where keep_matrix[j,s] == 0]  # 16 shadows that DIDN'T see j

    in_losses = [loss from each IN shadow]    # 16 values
    out_losses = [loss from each OUT shadow]  # 16 values

    # Fit Gaussian distributions
    μ_in, σ_in = mean(in_losses), std(in_losses)
    μ_out, σ_out = mean(out_losses), std(out_losses)

    # Get target's loss
    target_loss = target_model.loss(sample_j)

    # Compute log-likelihood ratio
    log_p_in = -log(σ_in) - 0.5 * ((target_loss - μ_in) / σ_in)²
    log_p_out = -log(σ_out) - 0.5 * ((target_loss - μ_out) / σ_out)²

    score = log_p_in - log_p_out

    # Higher score → target's loss matches IN pattern → likely MEMBER
```

**Intuition:**
1. Learn what "saw sample" vs "didn't see sample" looks like from 32 shadows
2. Check if target's loss matches the "saw it" pattern or "didn't see it" pattern

**Results:** 32 shadows: **64.0%** TPR @ 1% FPR

**Run:**
```bash
# Step 1: Generate shadow models (one-time, ~5-8 hours)
jupyter notebook prepare_and_train_shadow_lira.ipynb
# Run all cells to create 32 shadow models

# Step 2: Run attack (~20 minutes)
jupyter notebook MIA_LiRA.ipynb
# Run all cells to compute scores and evaluate
```

---

## Quick Start

### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch transformers datasets peft scikit-learn matplotlib tqdm numpy scipy
```

### Run Z-Score Attack
```bash
# If shadow models not trained yet:
jupyter notebook prepare_shadow_models.ipynb  # One-time setup

# Option 1: Z-Score with 32 shadows
jupyter notebook "MIA_zscore 32.ipynb"
# Expected: 73.0% TPR @ 1% FPR

# Option 2: Z-Score with 62 shadows (best performance)
jupyter notebook "MIA_zscore.ipynb"
# Expected: 77.9% TPR @ 1% FPR
```

### Run LiRA Attack
```bash
# If shadow models not trained yet:
jupyter notebook prepare_and_train_shadow_lira.ipynb  # ~5-8 hours

# Run LiRA Gaussian attack:
jupyter notebook MIA_LiRA.ipynb  # ~20 minutes
# Expected: 64.0% TPR @ 1% FPR
```

---

## Configuration

### Target Model Training
```python
Model: GPT-2 (124M)
LoRA: r=32, alpha=64, dropout=0.05
Training: 3 epochs, lr=2e-4, batch_size=8
Data: WikiText-103 + train_finetune.json
```

### Shadow Models (LiRA)
```python
Model: GPT-2 (124M)
LoRA: r=32, alpha=64, dropout=0.05
Training: 3 epochs, lr=2e-4, batch_size=8
Data: WikiText-103 only (10k samples per shadow)
Count: 32 shadows
```

### Keep Matrix (Membership Assignment)
```python
Shape: [2000 samples, 32 shadows]
Each sample: exactly 16 IN shadows, 16 OUT shadows
Random assignment per sample (proper per-example LiRA)
```

---

## Results

| Method | Shadows | TPR @ 1% FPR | File |
|--------|---------|--------------|------|
| LiRA (Gaussian) | 32 | **64.0%** | `MIA_LiRA.ipynb` |
| Z-Score | 32 | **73.0%** | `MIA_zscore 32.ipynb` |
| Z-Score | 62 | **77.9%** ✅ | `MIA_zscore.ipynb` |

**Key Findings:**
- Z-Score outperforms LiRA (77.9% vs 64.0%)
- More shadows improves Z-Score (62 shadows: 77.9% vs 32 shadows: 73.0%)
- Gaussian LiRA works but Z-Score's direct comparison is more effective

---

## Key Formulas

### Z-Score
```python
z = (shadow_loss - target_loss) / shadow_std
```

### Gaussian Log-Likelihood
```python
log P(x | μ, σ²) = -log(σ) - 0.5 * ((x - μ) / σ)²
```

### LiRA Score
```python
LLR = log P(target_loss | IN_dist) - log P(target_loss | OUT_dist)
```

---

## References

- Original LiRA paper: Carlini et al. (2022) "Membership Inference Attacks From First Principles"
- Course repo: https://github.com/2020pyfcrawl/18734-17731_Project_Phase2_3

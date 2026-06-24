# Mechanical Property Predictor

## Motivation

The current RL base version mainly uses structural proxy rewards, such as collision penalty, hydrogen-bond changes, and local RMSD. These signals are useful for structural sanity, but they are still only approximations of mechanical performance.

This subproject builds a direct mechanical-property predictor that can later be used as the main terminal reward model for RL-guided mechanical-protein optimization.

## Base Version Scope

### Model

```text
protein sequence
  -> ESM2 pooled embedding
  -> shared MLP backbone
  -> strength head
  -> toughness head
```

### Input

Version 1 uses protein sequence only.

### Output

The model predicts two targets:

```text
strength  = v128, maximum tensile strength
toughness = v127, toughness averaged per amino acid
```

### Dataset

Default CSV path:

```text
/mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv
```

Required columns:

```text
Sequence
v127
v128
```

Optional ID column:

```text
PDB_ID
```

Current dataset quick check:

```text
rows: 7041
valid rows: 7041
sequence length range: 27..586
v127 range: 122.97..1499.5
v128 range: 266.2141..11108.0789
```

`v128` has a long tail, so RMSE can be dominated by extreme high-strength samples. The training code standardizes targets using train-set mean and standard deviation.

## Directory Structure

Implemented under:

```text
model/reward_module/mechanical-properties-predictor/
```

```text
mechanical-properties-predictor/
├── train.py
├── dataset_module
│   ├── __init__.py
│   └── dataset.py
├── metric_module
│   ├── __init__.py
│   └── metrics.py
├── model_module
│   ├── __init__.py
│   └── predictor.py
└── logging_module
    ├── __init__.py
    └── logger.py
```

## Module Responsibilities

### dataset_module

Responsibilities:

- load the mechanical-property CSV;
- validate sequence and target columns;
- build random train/validation/test splits;
- build sequence-similarity group splits using k-mer Jaccard clustering;
- compute and cache ESM2 pooled embeddings;
- construct PyTorch datasets.

Main APIs:

```python
load_mechanical_property_records(...)
build_split(...)
encode_records(...)
MechanicalPropertyDataset
TargetNormalizer
```

Split methods:

```text
random
similarity
```

The similarity split is a lightweight base-version implementation using greedy k-mer Jaccard grouping. For stricter publication-grade sequence-identity splits, this can later be replaced with MMseqs2 or CD-HIT.

### metric_module

Implemented metrics:

- R2
- MAE
- RMSE
- Spearman correlation
- top-5% hit rate
- top-10% hit rate
- top-5% enrichment factor
- top-10% enrichment factor
- OOD performance through the similarity-split test set

Metrics are reported separately for:

```text
strength
toughness
```

and also as mean summaries:

```text
mean/r2
mean/mae
mean/rmse
mean/spearman
```

### model_module

Implemented model:

```python
MechanicalPropertyMLP
```

Architecture:

```text
input_dim = ESM2 embedding dimension
hidden MLP layers
shared representation
strength_head
toughness_head
```

Default:

```text
input_dim = 1280
hidden_dims = 512,256
dropout = 0.1
```

### logging_module

Responsibilities:

- write epoch history to JSONL;
- write split indices;
- write final metrics;
- write train/val/test prediction CSV files;
- save best and last checkpoints;
- optionally write TensorBoard scalars;
- generate basic training plots.

## Training

### Recommended Random Split Training

```bash
conda activate mprl-vgpt

python model/reward_module/mechanical-properties-predictor/train.py \
  --csv-path /mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv \
  --split-method random \
  --output-dir outputs/mechanical_property_predictor/random_esm2_1280 \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --enable-tensorboard
```

### Sequence-Similarity Split Training

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --csv-path /mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv \
  --split-method similarity \
  --similarity-threshold 0.5 \
  --kmer-size 5 \
  --output-dir outputs/mechanical_property_predictor/similarity_esm2_1280 \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --enable-tensorboard
```

### Smoke Test

Use a small sample to verify code paths without running the full ESM2 embedding pass:

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --max-samples 32 \
  --split-method random \
  --output-dir /tmp/mprl-mechanical-predictor-smoke \
  --embedding-dim 1280 \
  --esm2-device cpu \
  --device cpu \
  --epochs 2 \
  --batch-size 8
```

This smoke test still loads ESM2. It is intended for functional validation, not performance.

## Important Arguments

### Data

```bash
--csv-path
--sequence-column Sequence
--toughness-column v127
--strength-column v128
--id-column PDB_ID
--max-samples
```

### Split

```bash
--split-method random
--split-method similarity
--val-fraction 0.1
--test-fraction 0.1
--similarity-threshold 0.5
--kmer-size 5
--seed 7
```

### ESM2

```bash
--embedding-dim 1280
--embedding-dim 2560
--embedding-dim 5120
--esm2-device auto
--esm2-batch-size 4
--embedding-cache-dir outputs/mechanical_property_predictor/embedding_cache
```

Embeddings are pooled by mean over residue tokens and cached as `.npy` files by sequence SHA1 hash.

### MLP

```bash
--hidden-dims 512,256
--dropout 0.1
```

### Optimization

```bash
--epochs 100
--batch-size 64
--learning-rate 1e-4
--weight-decay 1e-4
--patience 20
--device auto
```

### Logging

```bash
--output-dir outputs/mechanical_property_predictor/run_name
--enable-tensorboard
--log-level INFO
```

## Outputs

Example output directory:

```text
outputs/mechanical_property_predictor/random_esm2_1280/
```

Expected files:

```text
run_config.json
checkpoints/
├── best_model.pt
└── last_model.pt
logs/
├── history.jsonl
├── metrics.json
├── train_indices.txt
├── val_indices.txt
├── test_indices.txt
├── train_predictions.csv
├── val_predictions.csv
└── test_predictions.csv
plots/
├── loss.png
├── val_mean_r2.png
├── val_mean_mae.png
├── val_mean_rmse.png
└── val_mean_spearman.png
tensorboard/
```

## Model Selection

The current checkpoint selection criterion is validation MSE on standardized targets.

Recommended model-quality checks:

1. Validation/test R2 should be positive and stable.
2. Spearman correlation should improve even when RMSE is affected by target outliers.
3. Top-5% and top-10% hit rates should be above random baseline.
4. Enrichment factor should be greater than 1.
5. Similarity-split test performance should be treated as OOD performance.

## How This Connects Back to RL

After a predictor is trained, the intended integration path is:

1. Load `best_model.pt`.
2. Encode candidate protein sequence with the same ESM2 model and pooling.
3. Predict:

```text
strength
toughness
```

4. Use predicted mechanical properties inside `TerminalRewardCalculator`.
5. Replace or augment current structural proxy reward with terminal mechanical reward.

Suggested scalar terminal reward:

```text
terminal_reward =
  w_strength  * normalized(predicted_strength)
  + w_toughness * normalized(predicted_toughness)
```

The predictor should be evaluated carefully before being used as the main RL reward, because reward hacking is likely if the model is inaccurate on mutated/OOD sequences.

## Current Boundaries

- Version 1 uses sequence-only input.
- ESM2 is frozen and used as a feature extractor.
- The MLP predicts two continuous properties.
- The similarity split is a lightweight k-mer approximation, not a strict sequence-identity split.
- No structure features are used yet.
- No uncertainty estimation is implemented yet.
- No calibration against RL-generated mutants has been done yet.

## Recommended Next Steps

1. Train random split model to verify the basic signal.
2. Train similarity split model to estimate OOD generalization.
3. Compare top-k enrichment for strength and toughness.
4. Inspect prediction errors for high-v128 outliers.
5. Add uncertainty estimation or ensemble models before RL integration.
6. Replace the placeholder terminal reward with this predictor once validation/OOD performance is acceptable.

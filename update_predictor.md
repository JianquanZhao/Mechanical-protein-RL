# Mechanical Property Predictor Updates

## 2026-06-24 Train Metrics Visualization and Log Analysis

### Context

Input log:

```text
train_predictor.log
```

The log corresponds to one run of the mechanical-property predictor:

```text
outputs/mechanical_property_predictor/random_esm2_1280
```

The run used the random split, ESM2 embeddings, and the MLP predictor.

### What Was Changed

Modified:

```text
model/reward_module/mechanical-properties-predictor/train.py
model/reward_module/mechanical-properties-predictor/logging_module/logger.py
```

#### 1. Added per-epoch train metrics

Before this change, each epoch logged:

```text
train_loss
val_loss
val metrics
```

This was not enough to diagnose overfitting because validation metrics existed, but train-side R2/MAE/RMSE/Spearman/top-hit metrics were only computed once at the end.

Now the training loop can evaluate train-set metrics during training.

New argument:

```bash
--train-metrics-every
```

Default:

```bash
--train-metrics-every 1
```

This records train metrics every epoch. Use `0` to disable per-epoch train metric evaluation if speed matters.

Logged train metrics include:

```text
train/strength/r2
train/strength/mae
train/strength/rmse
train/strength/spearman
train/strength/top_5pct_hit_rate
train/strength/top_10pct_hit_rate
train/strength/top_5pct_enrichment_factor
train/strength/top_10pct_enrichment_factor

train/toughness/r2
train/toughness/mae
train/toughness/rmse
train/toughness/spearman
train/toughness/top_5pct_hit_rate
train/toughness/top_10pct_hit_rate
train/toughness/top_5pct_enrichment_factor
train/toughness/top_10pct_enrichment_factor

train/mean/r2
train/mean/mae
train/mean/rmse
train/mean/spearman
```

#### 2. Improved TensorBoard logging

TensorBoard now writes both generic and split-aware scalar names.

Examples:

```text
metrics/train/mean/r2
metrics/val/mean/r2
train/mean/r2
val/mean/r2
train/strength/r2
val/strength/r2
train/toughness/spearman
val/toughness/spearman
```

This makes it easier to compare train and validation curves directly in TensorBoard.

Run TensorBoard with:

```bash
tensorboard --logdir outputs/mechanical_property_predictor/random_esm2_1280/tensorboard
```

#### 3. Fixed and expanded metric plots

The previous `plot_history()` looked for keys such as:

```text
mean/r2
mean/mae
```

But epoch records actually used prefixed keys such as:

```text
val/mean/r2
val/mean/mae
```

Therefore some metric plots could be missing.

Now plots include train/val curves when available:

```text
plots/loss.png
plots/mean_r2.png
plots/mean_mae.png
plots/mean_rmse.png
plots/mean_spearman.png
plots/strength_r2.png
plots/strength_mae.png
plots/strength_rmse.png
plots/strength_spearman.png
plots/toughness_r2.png
plots/toughness_mae.png
plots/toughness_rmse.png
plots/toughness_spearman.png
```

Sparse train metrics are supported. If `--train-metrics-every` is larger than 1, missing points are skipped during plotting.

### Verification

Compilation check:

```bash
python -m py_compile \
  model/reward_module/mechanical-properties-predictor/train.py \
  model/reward_module/mechanical-properties-predictor/logging_module/logger.py
```

Result:

```text
passed
```

No full retraining was run in this update, because full ESM2 embedding and training are expensive. The changes are logging/evaluation-path changes and were validated syntactically.

## Analysis of `train_predictor.log`

### Run Summary

The log shows:

```text
records: 7041
epochs run: 31
early stopping: triggered at epoch 31
best epoch: 11
best val_loss: 1.549209
```

Training stopped because validation loss did not improve for the configured patience window.

### Key Metrics at Best Epoch

Best epoch:

```text
epoch = 11
train_loss = 0.277586
val_loss = 1.549209
```

Validation metrics at best epoch:

```text
mean/r2        = 0.388471
mean/mae       = 55.2688
mean/rmse      = 230.438
mean/spearman  = 0.805564

strength/r2        = 0.119183
strength/mae       = 70.9547
strength/rmse      = 408.224
strength/spearman  = 0.780139

toughness/r2        = 0.657760
toughness/mae       = 39.5829
toughness/rmse      = 52.6522
toughness/spearman  = 0.830989
```

Final best-checkpoint train/val/test metrics:

```text
train mean/r2 = 0.762777
val   mean/r2 = 0.388471
test  mean/r2 = 0.539446

train strength/r2 = 0.747046
val   strength/r2 = 0.119183
test  strength/r2 = 0.476351

train toughness/r2 = 0.778508
val   toughness/r2 = 0.657760
test  toughness/r2 = 0.602541
```

### Diagnosis 1: Clear Overfitting

Evidence:

```text
train_loss:
  epoch 1  = 0.715059
  epoch 11 = 0.277586
  epoch 31 = 0.099099

val_loss:
  epoch 1  = 1.660203
  epoch 11 = 1.549209
  epoch 31 = 1.587900
```

The model keeps fitting the training set after epoch 11, but validation loss does not continue improving.

This means the current MLP capacity is enough to memorize training-set patterns, but generalization saturates early.

Recommended changes:

1. Increase regularization:

```bash
--dropout 0.2
--weight-decay 1e-3
```

2. Reduce model size:

```bash
--hidden-dims 256,128
```

3. Lower patience:

```bash
--patience 8
```

4. Add learning-rate scheduling in a future update:

```text
ReduceLROnPlateau on val_loss
```

### Diagnosis 2: Strength Head Generalizes Poorly

The most important issue is the strength target:

```text
val strength/r2 = 0.119183
val strength/rmse = 408.224
```

By contrast, toughness is much better:

```text
val toughness/r2 = 0.657760
val toughness/rmse = 52.6522
```

This suggests:

- sequence embedding contains useful information for toughness;
- strength (`v128`) is harder and likely affected by extreme outliers;
- the long tail in `v128` is hurting RMSE and R2;
- a shared MSE loss may not be ideal for both targets.

Recommended changes:

1. Try robust target transform for `v128`:

```text
log1p(strength)
```

2. Try robust loss:

```text
HuberLoss / SmoothL1Loss
```

3. Add per-target loss weights:

```text
strength_loss_weight
toughness_loss_weight
```

4. Analyze high-v128 outliers using `test_predictions.csv`.

### Diagnosis 3: Ranking Signal Is Stronger Than Calibration

Even though strength R2 is low, Spearman is high:

```text
val strength/spearman = 0.780139
val toughness/spearman = 0.830989
```

Top-k enrichment is also strong:

```text
val strength/top_10pct_enrichment_factor = 6.05634
val toughness/top_10pct_enrichment_factor = 5.77465
```

This means the model is better at ranking candidates than predicting exact physical values.

For RL reward use, this is important:

- if RL uses raw predicted values, calibration matters;
- if RL uses ranking or normalized reward, the current predictor may already provide useful direction;
- however, low strength R2 means raw strength predictions should not yet be trusted as absolute values.

Recommended RL integration style:

```text
Use normalized or rank-like reward first.
Avoid using raw v128 predictions directly as unbounded reward.
```

### Diagnosis 4: Random Split May Be Too Optimistic

The current run appears to use:

```text
split_method = random
```

Random split can place highly similar sequences across train/val/test. Therefore the reported Spearman and enrichment may be optimistic for RL-generated mutants or new protein families.

Recommended next experiment:

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --split-method similarity \
  --similarity-threshold 0.5 \
  --kmer-size 5 \
  --output-dir outputs/mechanical_property_predictor/similarity_esm2_1280 \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --dropout 0.2 \
  --weight-decay 1e-3 \
  --enable-tensorboard
```

This will estimate OOD performance more honestly.

### Diagnosis 5: ESM2 Embedding Cache Is Working

The log shows ESM2 encoding progressed to:

```text
Encoded ESM2 batch 7041/7041
```

This means all records were encoded successfully. Reusing the same `--embedding-cache-dir` should make future runs much faster.

Recommended:

```bash
--embedding-cache-dir outputs/mechanical_property_predictor/esm2_1280_cache
```

Use the same cache for random/similarity split comparisons.

## Recommended Next Training Plan

### Experiment A: Regularized Random Split

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --split-method random \
  --output-dir outputs/mechanical_property_predictor/random_esm2_1280_reg \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --hidden-dims 256,128 \
  --dropout 0.2 \
  --weight-decay 1e-3 \
  --epochs 100 \
  --patience 10 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --enable-tensorboard
```

Goal:

```text
Reduce train/val generalization gap.
```

### Experiment B: Similarity Split OOD Test

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --split-method similarity \
  --similarity-threshold 0.5 \
  --kmer-size 5 \
  --output-dir outputs/mechanical_property_predictor/similarity_esm2_1280_reg \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --hidden-dims 256,128 \
  --dropout 0.2 \
  --weight-decay 1e-3 \
  --epochs 100 \
  --patience 10 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --enable-tensorboard
```

Goal:

```text
Estimate OOD performance before using predictor as RL reward.
```

### Experiment C: Strength-Target Robustness

Future code improvement:

```text
--target-transform none/log1p
--loss mse/huber
--strength-loss-weight
--toughness-loss-weight
```

Reason:

```text
v128 has strong outliers and currently has weak validation R2.
```

## Current Judgment

The predictor is already learning useful sequence-property signal, especially ranking signal and toughness prediction.

However, it is not yet strong enough to be used as the sole RL reward because:

1. train/val gap is large;
2. strength calibration is weak;
3. only random split has been tested;
4. OOD behavior is unknown;
5. target outliers likely distort MSE training.

The next milestone should be:

```text
Obtain acceptable similarity-split performance and stabilize strength prediction.
```

Only after that should the predictor be connected into `TerminalRewardCalculator` as the main mechanical-property reward.

## 2026-06-24: log1p Target Transform and Huber Loss

### Modification Basis

The data analysis showed that both mechanical targets are positive continuous values, and `v128` has a much stronger long-tail distribution than `v127`. Under direct MSE optimization, a small number of very high-strength samples can dominate the gradient and make the model less stable for the majority of samples.

Therefore, the training target pipeline has been changed to:

```text
raw v127/v128
  -> log1p target transform
  -> train-set target standardization
  -> model training
```

During evaluation and prediction export, outputs are inverse-transformed back to the original `v127`/`v128` scale:

```text
normalized prediction
  -> inverse standardization
  -> expm1
  -> raw-scale metrics and prediction CSV
```

### Code Changes

Updated file:

```text
model/reward_module/mechanical-properties-predictor/train.py
```

Main changes:

1. Added `--target-transform`, with choices `none` and `log1p`; the default is `log1p`.
2. Added `--loss`, with choices `mse` and `huber`; the default is `huber`.
3. Added `--huber-beta`, default `1.0`.
4. Added `transform_targets(...)` and `inverse_transform_targets(...)`.
5. Changed target normalization from raw-target standardization to transformed-target standardization.
6. Changed default training criterion from `nn.MSELoss()` to `nn.SmoothL1Loss(beta=--huber-beta)`.
7. Changed evaluation so MAE, RMSE, R2, Spearman, hit-rate, enrichment, and prediction CSV files are computed on the original raw target scale.
8. Saved `target_transform`, `loss`, and `huber_beta` into `run_config.json`, checkpoints, and final metrics.

Updated documentation:

```text
mechanical-property-predictor.md
```

The training documentation now states that the default predictor training uses `log1p` target transform plus Huber loss, while reported metrics remain on the original target scale.

### Recommended Training Command

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --csv-path /mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv \
  --split-method random \
  --output-dir outputs/mechanical_property_predictor/random_esm2_1280_log1p_huber \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --target-transform log1p \
  --loss huber \
  --huber-beta 1.0 \
  --enable-tensorboard
```

### Ablation Commands

To compare against the previous behavior:

```bash
--target-transform none --loss mse
```

To test only the target transform:

```bash
--target-transform log1p --loss mse
```

To test only robust regression:

```bash
--target-transform none --loss huber --huber-beta 1.0
```

### Expected Effect

This change should reduce sensitivity to extreme high-strength outliers and make validation/test metrics, especially `strength` RMSE and R2, more stable. It may also slightly reduce raw-scale performance on the rare largest `v128` samples, so top-target ranking metrics and high-target residual plots should still be checked after training.

## 2026-06-24: Target-Specific Loss Weights

### Modification Basis

After applying target transform, sequence-similarity split, and Huber loss, the predictor showed clear improvement, but the remaining optimization bottleneck is mainly controlled by the loss signal. Because the model output order is:

```text
[strength, toughness]
```

the training code now supports independent loss weights for the strength and toughness heads. This makes it possible to increase the strength gradient signal without changing the model architecture or the target preprocessing pipeline.

### Code Changes

Updated file:

```text
model/reward_module/mechanical-properties-predictor/train.py
```

Main changes:

1. Added `--strength-loss-weight`, default `1.0`.
2. Added `--toughness-loss-weight`, default `1.0`.
3. Added `WeightedRegressionLoss`, which supports both `mse` and `huber` base losses with per-target weighting.
4. The weighted loss is normalized by the sum of target weights, so changing weights changes the relative strength/toughness gradient contribution without mechanically scaling the whole loss by the absolute weight sum.
5. The same weighted criterion is used for training loss, validation loss, early stopping, checkpoint selection, and final split evaluation.
6. Saved `strength_loss_weight` and `toughness_loss_weight` into `run_config.json`, checkpoints, and final metrics.

Updated documentation:

```text
mechanical-property-predictor.md
```

The training documentation now records the default target-loss weights and clarifies that the target order is `strength, toughness`.

### Recommended Strength-Enhanced Training Command

```bash
python model/reward_module/mechanical-properties-predictor/train.py \
  --csv-path /mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv \
  --split-method similarity \
  --similarity-threshold 0.5 \
  --kmer-size 5 \
  --output-dir outputs/mechanical_property_predictor/similarity_esm2_1280_log1p_huber_strength2 \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --target-transform log1p \
  --loss huber \
  --huber-beta 1.0 \
  --strength-loss-weight 2.0 \
  --toughness-loss-weight 1.0 \
  --enable-tensorboard
```

### Suggested Ablation

Recommended first comparison:

```text
1. strength=1.0, toughness=1.0
2. strength=2.0, toughness=1.0
3. strength=3.0, toughness=1.0
```

Primary metrics to watch:

```text
strength/r2
strength/rmse
strength/spearman
strength/top_5pct_hit_rate
toughness/r2
toughness/spearman
mean/r2
```

If strength improves while toughness collapses, reduce `--strength-loss-weight` or increase `--toughness-loss-weight`. If both improve, the previous training objective was under-weighting the strength head. If only validation loss improves but raw-scale strength metrics do not, the weighted normalized loss is not aligned enough with the raw-scale mechanical-property objective and should be paired with target-range or top-k focused analysis.

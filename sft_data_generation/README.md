# Mechanical Protein SFT Pipeline

This directory implements an offline, versioned pipeline for generating
mutation-action labels and pretraining the existing per-residue DDQN Q head.
The online RL code does not import this package.

## Boundaries

- `sft_data_generation/adapters/rl_contract.py` is the only offline module that
  imports `model.*`. It reuses the exact environment mutation, local repack,
  step reward and seven-feature random-forest terminal predictor.
- CPU generation does not load ESM2. ESM2/SFT proposals can be supplied later
  as a candidate table, while labels always come from PyRosetta plus the
  mechanical predictor.
- `train_sft.py` consumes only a validated dataset release. It freezes ESM2 and
  trains a Q head with the same `1280 -> 256 -> 256 -> 20` residue-wise layout
  as DDQN.

## Storage Layout

```text
/mnt/nas/jianquanzhao/data/mprl/SFT/
├── data/source/       # materialized WT cleaned PDBs
├── data/generated/    # manifests, tasks, replicates, labels and releases
├── model/<version_timestamp>/
├── log/<version_timestamp>/
└── tensorboard/<version_timestamp>/
```

## Environments

On a CPU data-generation server:

```bash
conda env create -f sft_data_generation/environment-cpu.yaml
conda activate mprl-sft-data
```

On a GPU SFT server:

```bash
conda env create -f sft_data_generation/environment-train.yaml
conda activate mprl-sft-train
```

For the complete workflow on one machine:

```bash
conda env create -f sft_data_generation/environment-all.yaml
conda activate mprl-sft
```

The essential CPU packages are Python 3.10, PyRosetta, NumPy, pandas, SciPy,
scikit-learn 1.6.1, joblib, PyYAML and MMseqs2. The scikit-learn pin matches
the version used to serialize `params/hbond_random_forest.joblib`; using 1.7.x
loads with an `InconsistentVersionWarning` and is not recommended for final
label generation. PyArrow is needed only for optional
Parquet tables. GPU training additionally needs PyTorch, fair-esm and
TensorBoard. PyRosetta is not required by `train_sft.py`.

## Prepare Source Data

The production config uses MMseqs2 similarity clustering at 30% identity and
80% coverage, then assigns entire clusters to SFT train/validation/test. The
existing RL validation index is reserved and never enters SFT generation.
Source sequences are read through the same environment/PyRosetta loading path
as RL. This deliberately costs more than PDB text parsing, but it excludes
residues that the loader removes and therefore keeps every offline action index
aligned with the online environment.

```bash
bash sft_data_generation/scripts/prepare_source.sh
```

Use `configs/smoke.yaml` for a tiny dependency-light exact-dedup smoke test:

```bash
SFT_CONFIG=sft_data_generation/configs/smoke.yaml \
  bash sft_data_generation/scripts/prepare_source.sh --limit 4
```

## Round-0 Full Scan

`run_round0.sh` runs deterministic shards on one machine. Each worker is a
separate process, and rerunning the command skips task IDs already present in
that shard's JSONL progress file.

```bash
SFT_WORKERS=8 SFT_SHARDS=64 bash sft_data_generation/scripts/run_round0.sh
```

The first pass should avoid saving every mutant PDB. After aggregation, select
the global beam and rescan only retained actions with
`--save-candidate-structures` before constructing the next-depth state table.

For deeper rounds, ESM2 and SFT proposal tables can be merged on the same
machine with deterministic random coverage. The merge validates masks,
deduplicates actions and enforces the total budget (64 by default):

```bash
python -m sft_data_generation.cli --config sft_data_generation/configs/base.yaml build-candidates \
  --states /path/to/depth_001_states.csv \
  --proposals /path/to/esm_proposals.csv \
  --proposals /path/to/sft_proposals.csv \
  --budget 64 --random-count 16
```

Each proposal table needs `state_id`, `action_index`, `proposal_source` and
`proposal_score`. If no model table is supplied, the command produces a fully
model-free, reproducible random candidate set instead of silently pretending
that ESM2 or SFT was used.

## Release and Train

After combining the state tables and action labels for the desired rounds:

```bash
python -m sft_data_generation.cli --config sft_data_generation/configs/base.yaml release \
  --states /path/to/all_states.csv \
  --actions /path/to/all_labels.csv \
  --release-name sft_actions_v001

bash sft_data_generation/scripts/train_sft.sh
```

The best checkpoint is selected by validation macro NDCG and written to
`model/<version_timestamp>/best.pt`. It contains Q-head weights, architecture,
amino-acid ordering, target semantics and validation metrics. It deliberately
does not modify or initialize DDQN automatically; A/B RL integration remains a
separate controlled experiment.

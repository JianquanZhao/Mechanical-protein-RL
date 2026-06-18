# Mechanical-protein-RL

A reinforcement-learning project for mechanical-protein sequence and structure optimization. The current base version uses PyRosetta to apply point mutations, local side-chain repacking, and local minimization. It evaluates structure-level step rewards and trains a masked Double DQN policy over variable-length protein structures.

The current version supports:

- folder-based training over multiple protein structures;
- automatic train/validation splitting and index management;
- filtering of DNA/RNA-only, non-protein, and backbone-incomplete structures;
- text-level PDB cleaning before PyRosetta loading;
- ESM2 per-residue observation encoding;
- a per-residue Q head for mixed-length protein replay;
- variable-length replay buffer collation with padded batches;
- dataset-style epoch/batch training schedules;
- single-device and single-machine multi-GPU DataParallel training;
- stdout logging, JSONL/CSV logs, checkpoints, and candidate PDB outputs.

## Project Layout

```text
.
├── training.py                         # Root training entry point
├── batch_training.sh                   # Example training script
├── model
│   ├── agent_module
│   │   └── ddqn_agent.py               # DDQN agent and per-residue Q network
│   ├── dataset_module
│   │   └── dataset.py                  # PDB discovery, filtering, train/val split, epoch batches
│   ├── encoding_module
│   │   └── esm2_encoder.py             # ESM2 per-residue observation encoder
│   ├── environment_module
│   │   └── environment.py              # PyRosetta RL environment
│   ├── logging_module
│   │   └── training_logger.py          # Training logs, plots, TensorBoard support
│   ├── replay_buffer_module
│   │   └── replay_buffer.py            # ReplayBuffer with variable-length padded batches
│   └── reward_module
│       ├── reward_calculators.py       # Step and terminal rewards
│       └── wild_type.pdb               # Example PDB
├── tests                               # Unit tests and PyRosetta smoke test
├── update_dataset.md                   # Dataset and training-entry development notes
└── update_ddqn.md                      # DDQN, ESM2, and variable-length replay development notes
```

## Environment Setup

A conda environment is recommended. The development and smoke-test environment used in this project is named `mprl-vgpt`.

```bash
conda env create -f environment.yaml
conda activate mprl-vgpt
```

`environment.yaml` includes the RosettaCommons conda channel and declares `pyrosetta` as a dependency. PyRosetta is license-restricted; make sure your installation and usage comply with the applicable license.

If PyRosetta cannot be installed from the conda channel in your environment, comment out the `pyrosetta` line in `environment.yaml`, create the environment, and then install a licensed wheel manually:

```bash
conda env create -f environment.yaml
conda activate mprl-vgpt
pip install /path/to/pyrosetta-*.whl
```

For CUDA-specific PyTorch builds, adjust the `torch` entry in `environment.yaml` according to the official PyTorch installation matrix for your driver and CUDA runtime.

Verify the environment:

```bash
python -c "import torch; import pyrosetta; import esm; print('ok')"
pytest -q
```

Reference full-test result for the current base version:

```text
93 passed, 1 skipped
```

## Data Preparation

The training entry point expects a folder of PDB-like structure files, not a single PDB file:

```bash
--pdb-dir /path/to/pdb_folder
```

`ProteinStructureDataset` recursively discovers:

```text
.pdb
.ent
```

It creates train/validation index files under the data folder by default:

```text
train_index.txt
val_index.txt
```

By default, dataset preprocessing filters out:

- structures with no canonical protein residues;
- DNA/RNA-only structures;
- structures with fewer canonical protein residues than `--min-protein-residues`;
- structures where more than `--max-missing-backbone-fraction` of canonical residues miss any of `N/CA/C/O`.

Relevant arguments:

```bash
--pdb-dir /path/to/pdb_folder
--train-index /optional/train_index.txt
--val-index /optional/val_index.txt
--val-fraction 0.1
--dataset-seed 7
--recreate-splits
--min-protein-residues 1
--max-missing-backbone-fraction 0.05
```

If index files already exist, they are read and revalidated. To recreate the split:

```bash
--recreate-splits
```

## Training

### Recommended: Dataset Epoch/Batch Training

```bash
conda activate mprl-vgpt

python training.py \
  --pdb-dir /path/to/pdb_folder \
  --mode single \
  --device cuda:0 \
  --epochs 10 \
  --train-batch-size 8 \
  --max-steps 100 \
  --observation-encoder esm2 \
  --embedding-dim 1280 \
  --output-dir outputs/ddqn_esm2_run
```

Meaning:

- `--epochs 10`: run 10 dataset-level training passes;
- `--episodes-per-epoch`: number of PDB episodes per epoch; defaults to the full training split size;
- `--train-batch-size 8`: group 8 PDB episodes into each dataset batch;
- each PDB in a dataset batch still resets the environment and runs one independent RL episode;
- replay buffer insertion and DDQN optimization remain step-level operations.

### Compatibility Mode: Fixed Total Episodes

If `--epochs` is not provided, training falls back to the legacy `--episodes` mode:

```bash
python training.py \
  --pdb-dir /path/to/pdb_folder \
  --mode single \
  --device cuda:0 \
  --episodes 500 \
  --max-steps 100 \
  --output-dir outputs/ddqn_legacy_run
```

### Single-Machine Multi-GPU

```bash
python training.py \
  --pdb-dir /path/to/pdb_folder \
  --mode multi \
  --gpu-ids 0,1,2,3 \
  --epochs 10 \
  --train-batch-size 8 \
  --observation-encoder esm2 \
  --output-dir outputs/ddqn_multi_gpu
```

Multi-GPU training currently wraps the online and target networks with `torch.nn.DataParallel`. Environment rollouts are still executed sequentially in one process.

## Key Training Arguments

### Environment and Structure Updates

```bash
--max-steps 100
--mutable-positions 38,39,40
--local-repack-radius 8.0
--no-repack
--no-minimize
--minimize-backbone
--prevent-revisit-positions
--continue-on-update-error
```

By default, mutable positions are all canonical amino-acid residues in the cleaned PyRosetta Pose. If you pass explicit `--mutable-positions`, be careful: PDB cleaning can change pose residue numbering.

### PDB Cleaning Before PyRosetta Load

Environment-side PDB cleaning is enabled by default to reduce failures such as:

```text
ERROR: too many tries in fill_missing_atoms!
```

Relevant arguments:

```bash
--no-clean-pdb-before-load
--keep-cleaned-pdbs
--cleaned-pdb-dir outputs/cleaned_pdb_debug
```

Default cleaning behavior:

- remove non-canonical residues such as `ACY`;
- remove canonical residues missing any of `N/CA/C/O`;
- remove `HETATM` records such as ligand, DNA/RNA, water, and ions;
- fail early if the missing-backbone fraction exceeds the configured threshold.

### Observation Encoder

Default one-hot observation:

```bash
--observation-encoder default
```

For mixed-length protein training, ESM2 is recommended:

```bash
--observation-encoder esm2
--embedding-dim 1280
--esm2-device auto
--esm2-mutable-only
```

Supported DDQN embedding dimensions:

```text
1280
2560
5120
```

For the base version, `1280` is the recommended starting point.

### DDQN and Replay

```bash
--hidden-dims 256,256
--gamma 0.99
--learning-rate 1e-4
--micro-batch-size 16
--gradient-accumulation-steps 4
--replay-warmup-size 1000
--replay-capacity 100000
--target-sync-interval 250
--epsilon-start 1.0
--epsilon-end 0.05
--epsilon-decay-steps 50000
--use-amp
```

Effective optimization batch size:

```text
micro_batch_size * gradient_accumulation_steps
```

When `--observation-encoder esm2` is used, the ReplayBuffer automatically enables variable-length padded collation.

### Missing-Atom Handling in Reward Calculation

Local RMSD uses a penalty strategy by default so long-running training is not interrupted by occasional missing atoms:

```bash
--rmsd-missing-atom-policy penalize
--rmsd-missing-penalty 5.0
--min-rmsd-atoms 3
```

Available policies:

```text
raise
skip_residue
penalize
```

Use `raise` when debugging dirty input structures. Use `penalize` for long mixed-quality training runs.

## Outputs

Default output directory:

```text
outputs/ddqn_base/
```

Typical structure:

```text
outputs/ddqn_base/
├── run_config.json
├── checkpoints/
│   ├── agent.pt
│   ├── agent_final.pt
│   ├── replay_buffer.npz
│   └── replay_buffer_final.npz
├── candidates/
│   └── episode_0000.pdb
└── logs/
    ├── validation.jsonl
    ├── step_records.jsonl
    ├── episode_records.jsonl
    └── optimization_records.jsonl
```

Logs include:

- episode reward;
- step reward;
- action and decoded mutation;
- action-mask statistics;
- source PDB;
- epoch and batch indices;
- epsilon;
- optimization loss, Q values, TD error, and gradient norm;
- validation greedy-rollout results.

## Validation and Tests

Run all tests:

```bash
pytest -q
```

Run the core tests that do not require PyRosetta:

```bash
pytest -q \
  tests/test_dataset.py \
  tests/test_environment.py \
  tests/test_reward_calculators.py \
  tests/test_replay_buffer.py \
  tests/test_ddqn_agent.py
```

Run the PyRosetta smoke test:

```bash
pytest -q tests/test_environment_pyrosetta_smoke.py
```

Minimal CPU smoke training:

```bash
conda run -n mprl-vgpt python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --epochs 1 \
  --episodes-per-epoch 1 \
  --train-batch-size 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --checkpoint-every 1 \
  --no-resume-logs \
  --validate-every 0 \
  --log-level WARNING
```

## Current Base-Version Boundaries

- `train-batch-size` is a PDB-episode scheduling batch, not a parallel environment batch.
- Environment rollouts are still sequential; vectorized environments or multiprocessing rollout workers can be added later.
- The terminal-reward predictor interface exists, but the real mechanical-property prediction model still needs to be connected and calibrated.
- PDB cleaning removes problematic residues; it does not reconstruct missing atoms.
- The default one-hot observation is not suitable for mixed-length protein training; use `--observation-encoder esm2` for mixed-length datasets.
- Non-standard amino acids and modified residues are not included in the default mutation space.

## Suggested First Training Checks

For the first training runs, monitor:

- whether episode reward increases or at least remains stable;
- whether replay-buffer warmup is followed by finite losses with no NaNs;
- whether invalid/no-op actions are correctly masked;
- whether candidate PDB files can be opened by PyRosetta and visualization tools;
- whether `local_rmsd_status` frequently becomes `penalized_missing_atoms`;
- whether train and validation rewards diverge;
- whether ESM2 mode has stable memory use, throughput, and padded-batch behavior.

## Related Documentation

- [analysis.md](analysis.md): project-level code analysis.
- [update_ddqn.md](update_ddqn.md): DDQN, ESM2, per-residue Q head, and variable-length replay notes.
- [update_dataset.md](update_dataset.md): dataset, PDB cleaning, and epoch/batch training notes.
- [README_DDQN_AGENT.md](README_DDQN_AGENT.md): DDQN agent details.
- [README_TRAINING_LOGGER.md](README_TRAINING_LOGGER.md): training logger details.
- [model/environment_module/README_ENVIRONMENT.md](model/environment_module/README_ENVIRONMENT.md): environment module details.
- [model/replay_buffer_module/README_REPLAY_BUFFER.md](model/replay_buffer_module/README_REPLAY_BUFFER.md): ReplayBuffer module details.

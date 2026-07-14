# Terminal Reward Update

## Idea

This update adds a structure-dependent terminal reward for mechanical protein RL.
The reward is based on the seven topology/hydrogen-bond features selected in the
previous statistical machine-learning analysis:

1. `sequence_length`
2. `hbond_per_residue`
3. `seq_class_nonlocal_per_residue`
4. `strong_nonlocal_fraction`
5. `strong_nonlocal_per_residue`
6. `nonlocal_backbone_backbone_per_residue`
7. `hbond_contact_order`

The terminal reward predicts two mechanical-property targets:

- `v127`: toughness averaged per amino acid
- `v128`: maximum tensile strength

The predictor is a random-split random forest model trained from the existing
hbond analysis table. Targets are modeled with `log1p` during training and
converted back to physical units during inference.

The reward module supports two use modes:

- single-structure reward from a PyRosetta terminal pose or a PDB file
- dual-structure reward from both the PyRosetta relaxed terminal structure and a
  separately predicted terminal structure

For dual-structure scoring, each structure is scored independently and then
combined by quality weights. Predicted-structure quality can be supplied as
pLDDT, and the relaxed PyRosetta structure can be supplied as a normalized
quality score. A disagreement penalty is subtracted when the two structures give
substantially different rewards.

## Code Organization

New terminal reward package:

- `model/reward_module/terminal_reward/features.py`
  - parses PDB files without requiring Biopython
  - extracts explicit-hydrogen hydrogen bonds, distances, angles and topology
    classes
  - computes the seven selected features
  - supports PyRosetta-like pose input by dumping the pose to a temporary PDB

- `model/reward_module/terminal_reward/model_artifact.py`
  - trains or loads the random-split random forest artifact
  - uses the existing random split index files from the hbond analysis outputs
  - stores target means/stds for z-score reward scalarization
  - default artifact path:
    `model/reward_module/terminal_reward/artifacts/hbond_random_forest.joblib`

- `model/reward_module/terminal_reward/calculator.py`
  - defines `HbondTopologyTerminalRewardCalculator`
  - defines `TerminalRewardScalarization`
  - provides `evaluate_pdb`, `evaluate_pose`, and `evaluate_dual_structures`

- `model/reward_module/terminal_reward/__init__.py`
  - exports the public terminal reward API

Modified reward module:

- `model/reward_module/reward_calculators.py`
  - re-exports the new hbond/topology terminal reward classes
  - keeps the previous `TerminalRewardCalculator` available
  - fixes the previous predictor wrapper so `_predict_one()` calls the supplied
    predictor instead of returning placeholder zeros

Smoke test:

- `model/reward_module/smoke_test_hbond_terminal_reward.py`
  - loads or trains the random forest artifact
  - scores one real PDB file
  - scores the same PDB as a dual-structure sanity check

## ColabFold Environment

Created and verified a ColabFold environment with bioconda:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/condabin/conda create -y \
  --solver libmamba \
  -p /mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda \
  -c conda-forge -c bioconda \
  python=3.10 colabfold=1.5.5
```

Activate it with:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/condabin/conda activate \
  /mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda
```

Verification commands:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/python - <<'PY'
import colabfold
print("colabfold", getattr(colabfold, "__version__", "import-ok"))
PY

/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/colabfold_batch --help
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/mmseqs version
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/hhsearch -h
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/kalign -h
```

Observed verification:

- `import colabfold` succeeded
- `colabfold_batch --help` printed the CLI usage
- `mmseqs version` returned `18.8cc5c`
- `hhsearch -h` returned HHsearch 3.3.0 help
- `kalign -h` returned Kalign 2.04 help

The first GitHub/PyPI installation attempts were affected by TLS/network
instability. The bioconda route with `--solver libmamba` completed successfully
after retrying the large `cudatoolkit-11.8.0` download.

## Test Examples

Compile check:

```bash
python -m compileall \
  model/reward_module/terminal_reward \
  model/reward_module/reward_calculators.py \
  model/reward_module/smoke_test_hbond_terminal_reward.py
```

New terminal reward smoke test:

```bash
python model/reward_module/smoke_test_hbond_terminal_reward.py
```

Observed result:

- single-structure reward was finite
- predicted strength and toughness were finite
- all seven features were extracted
- dual-structure reward path was finite

Previous terminal reward wrapper smoke test:

```bash
python model/reward_module/smoke_test_terminal_reward.py
```

Observed result:

- the wrapper returned the expected scalar reward
- the supplied dummy predictor was called correctly

## Usage

Single terminal PyRosetta pose:

```python
from model.reward_module.reward_calculators import (
    HbondTopologyTerminalRewardCalculator,
    TerminalRewardScalarization,
)

terminal_reward = HbondTopologyTerminalRewardCalculator(
    scalarization=TerminalRewardScalarization(
        strength_weight=1.0,
        toughness_weight=1.0,
        disagreement_penalty=0.25,
    )
)

result = terminal_reward.evaluate_pose(current_pose)
reward = result.reward
```

Dual terminal structures:

```python
dual_result = terminal_reward.evaluate_dual_structures(
    relaxed_pose=current_pose,
    predicted_pdb_path="predicted_terminal_structure.pdb",
    relaxed_quality=0.9,
    predicted_plddt=85.0,
)

reward = dual_result.reward
```

Use inside the RL environment by passing the calculator as the terminal reward
calculator:

```python
env = MechanicalProteinEnv(
    ...,
    terminal_reward_calculator=terminal_reward,
)
```

## Notes

- The hbond feature extractor is designed for PDB structures with explicit
  hydrogens. If no hydrogens are present, the module will still run, but the
  hydrogen-bond count features will be near zero and a warning will be recorded.
- pLDDT is read from B-factors only when the values look like confidence scores
  in `[1, 100]`. Ordinary crystallographic B-factors near zero are treated as
  unknown quality rather than low-confidence predicted structures.
- The current terminal reward is a learned proxy for simulated mechanical
  properties, not a replacement for molecular mechanics or pulling simulation.
  It is most useful as a terminal reward guide and should be monitored together
  with raw predicted `strength`, raw predicted `toughness`, feature diagnostics,
  and structure-quality weights.

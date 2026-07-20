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

## 2026-07-14 Terminal Reward Recovery Check

### Context

Because of an incorrect Git operation, some newly added terminal reward code may
have been deleted or reverted. The requested model artifact path is:

```text
params/hbond_random_forest.joblib
```

### Check Result

The terminal reward Python package still exists:

```text
model/reward_module/terminal_reward/__init__.py
model/reward_module/terminal_reward/features.py
model/reward_module/terminal_reward/model_artifact.py
model/reward_module/terminal_reward/calculator.py
```

The integration in `model/reward_module/reward_calculators.py` also still
exists. The file imports and re-exports:

```text
HbondTopologyTerminalRewardCalculator
TerminalRewardScalarization
StructureRewardResult
DualStructureTerminalRewardResult
HbondFeatureExtractor
HbondFeatureResult
```

The previous generic `TerminalRewardCalculator` wrapper was also checked. Its
`_predict_one(...)` method still calls `_run_predictor(...)`, so it has not
fallen back to the old placeholder-zero behavior.

### Issue Found

`params/hbond_random_forest.joblib` existed, but it was only a 134-byte Git LFS
pointer:

```text
version https://git-lfs.github.com/spec/v1
size 142555840
```

Therefore `joblib.load("params/hbond_random_forest.joblib")` failed before the
fix. The hbond analysis table and random split index files were still available
under:

```text
outputs/mechanical_property_predictor/hbond_analysis/
```

### Code Changes

Updated:

```text
model/reward_module/terminal_reward/model_artifact.py
```

Changes:

1. Changed the default model artifact path to:

```text
params/hbond_random_forest.joblib
```

2. Changed the default hbond analysis directory to the current existing output
   directory:

```text
outputs/mechanical_property_predictor/hbond_analysis/
```

3. Added Git LFS pointer detection. If the artifact is a small LFS pointer and
   the hbond analysis table exists, the code retrains the random forest artifact
   and writes a usable local `joblib` file to `params/hbond_random_forest.joblib`.
   If the table is missing, the error message now explicitly asks for
   `git lfs pull` or restoration of the hbond analysis table.

Updated:

```text
model/reward_module/smoke_test_hbond_terminal_reward.py
```

Changes:

1. The smoke test now explicitly uses `DEFAULT_ARTIFACT_PATH`, which points to
   `params/hbond_random_forest.joblib`.
2. The smoke test adds the repository root to `sys.path`, so the model artifact
   can be loaded correctly even when the script is executed directly as:

```bash
python model/reward_module/smoke_test_hbond_terminal_reward.py
```

### Artifact Recovery

The local artifact was rebuilt from:

```text
outputs/mechanical_property_predictor/hbond_analysis/hbond_length_disentanglement_table.csv
outputs/mechanical_property_predictor/hbond_analysis/dataset_random_train_pdb_ids.txt
outputs/mechanical_property_predictor/hbond_analysis/dataset_random_val_pdb_ids.txt
outputs/mechanical_property_predictor/hbond_analysis/dataset_random_test_pdb_ids.txt
```

Recovered artifact:

```text
params/hbond_random_forest.joblib
```

Recovered size:

```text
142555850 bytes
```

Recovered selected features:

```text
sequence_length
hbond_per_residue
seq_class_nonlocal_per_residue
strong_nonlocal_fraction
strong_nonlocal_per_residue
nonlocal_backbone_backbone_per_residue
hbond_contact_order
```

Recovered random-split sizes:

```text
train = 5632
val   = 704
test  = 705
```

Recovered test metrics:

```text
toughness_r2   = 0.657354
toughness_mae  = 38.638565
toughness_rmse = 63.391846
strength_r2    = 0.306388
strength_mae   = 65.211485
strength_rmse  = 180.179298
```

### Verification

Compile check:

```bash
python -m compileall \
  model/reward_module/terminal_reward \
  model/reward_module/reward_calculators.py \
  model/reward_module/smoke_test_hbond_terminal_reward.py
```

Result:

```text
passed
```

Terminal reward smoke test:

```bash
python model/reward_module/smoke_test_hbond_terminal_reward.py
```

Result:

```text
HbondTopologyTerminalRewardCalculator smoke test passed.
```

The smoke test verified:

- `params/hbond_random_forest.joblib` exists and is loadable;
- single-structure reward is finite;
- predicted strength and toughness are finite;
- all seven hbond/topology features are extracted;
- dual-structure reward path is finite.

Previous terminal reward wrapper smoke test:

```bash
python model/reward_module/smoke_test_terminal_reward.py
```

Result:

```text
TerminalRewardCalculator smoke test passed.
```

### Conclusion

The terminal reward code was mostly still present. The main problem was that the
new model artifact path contained only a Git LFS pointer rather than the real
`joblib` file. The default artifact path and hbond analysis data path have now
been aligned with the current repository layout, the model artifact has been
rebuilt into `params/hbond_random_forest.joblib`, and both new and previous
terminal reward smoke tests pass.

## 2026-07-15: predicted-structure evaluation workflow

### Goal

Test the external pipeline:

```text
spider silk sequence -> ColabFold predicted structure -> hbond/topology features -> mechanical-property prediction -> correlation with measured mechanical properties
```

### Code

Created and executed:

```text
model/reward_module/terminal_reward/evaluate_based_predicted.ipynb
```

The notebook implements:

- parsing measured toughness and tensile strength from `/home/jianquanzhao/git_reps/spider_silk_codes/mechanical_properties.csv`;
- parsing FASTA sequences from `/home/jianquanzhao/git_reps/spider_silk_codes/raw_data/`;
- matching FASTA files to measured samples by the second underscore-separated FASTA filename token, which corresponds to `idv_id`;
- preparing ColabFold input FASTA files under `output/evaluate-mechanical-predictor/pdb-colab/input-fasta/`;
- indexing predicted PDB files under `output/evaluate-mechanical-predictor/pdb-colab/results/`;
- applying `HbondTopologyTerminalRewardCalculator` to predicted PDB files when they are available;
- computing Pearson and Spearman correlations at sequence level and idv-level mean aggregation;
- writing tables, figures, and a JSON run summary.

### Current data parsing result

The notebook successfully parsed and matched:

```text
mechanical property rows:                 446
rows with toughness and tensile strength: 393
matched idv samples:                      268
matched FASTA sequences:                  3513
sequence length range:                    79 to 1854 aa
mean sequence length:                     490.33 aa
```

For a controlled smoke test, the notebook selected the three shortest matched sequences with length <= 180 aa as ColabFold input examples.

### Outputs

Generated tables:

```text
output/evaluate-mechanical-predictor/tables/matched_sequences.csv
output/evaluate-mechanical-predictor/tables/colabfold_input_sequences.csv
output/evaluate-mechanical-predictor/tables/colabfold_predicted_pdb_index.csv
output/evaluate-mechanical-predictor/tables/predicted_mechanical_properties.csv
output/evaluate-mechanical-predictor/tables/prediction_correlations.csv
```

Generated figures:

```text
output/evaluate-mechanical-predictor/figures/measured_property_distributions.png
output/evaluate-mechanical-predictor/figures/matched_sequence_length_distribution.png
output/evaluate-mechanical-predictor/figures/prediction_status_no_pdb.png
```

Run summary:

```text
output/evaluate-mechanical-predictor/evaluation_summary.json
```

### ColabFold validation status

The ColabFold CLI exists at:

```text
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda/bin/colabfold_batch
```

A smoke prediction was attempted with:

```bash
colabfold_batch \
  --data output/evaluate-mechanical-predictor/colabfold-data \
  --msa-mode single_sequence \
  --num-models 1 \
  --num-recycle 1 \
  --num-relax 0 \
  --overwrite-existing-results \
  output/evaluate-mechanical-predictor/pdb-colab/smoke-input \
  output/evaluate-mechanical-predictor/pdb-colab/smoke-result
```

Observed issues:

- the default ColabFold cache tried to write under `/home/jianquanzhao/.cache/colabfold`, which is read-only in the current execution sandbox;
- using `--data output/evaluate-mechanical-predictor/colabfold-data` fixed the writable cache path;
- the first run then needed to download AlphaFold/ColabFold parameters of about 3.47 GB;
- the sandboxed run could not access Google Storage;
- the escalated run started the download, but the transfer was slow and was interrupted before completion;
- no predicted PDB file was generated in this run.

Because no predicted PDB files were available, the structure-based predictor did not produce property predictions yet, and Pearson/Spearman correlations remain unavailable in the current run.

### Interpretation

The data parsing and evaluation scaffold are ready. The current blocker is not the hbond/topology mechanical-property predictor, but the missing ColabFold model parameters and therefore missing predicted structures.

The next complete evaluation should:

1. finish the ColabFold parameter download into `output/evaluate-mechanical-predictor/colabfold-data`;
2. run ColabFold first on the three prepared smoke sequences;
3. rerun notebook sections that index PDBs, compute terminal reward predictions, and calculate correlations;
4. consider protonating ColabFold PDBs before hbond-feature extraction, because ColabFold PDBs usually lack explicit hydrogens while the hbond predictor relies on hydrogen-bond topology features.

## 2026-07-15: local ColabFold parameter run

### Parameter source

The user provided an existing local AlphaFold/ColabFold parameter directory:

```text
/mnt/nas/jianquanzhao/afdb/params
```

To avoid ColabFold triggering a new 3.47 GB download, the evaluation notebook now creates a lightweight local data directory:

```text
output/evaluate-mechanical-predictor/colabfold-local-data/params
```

This directory stores symlinks to the local parameter files and a local `download_finished.txt` marker. The notebook default `COLABFOLD_DATA_DIR` was updated to:

```text
output/evaluate-mechanical-predictor/colabfold-local-data
```

### ColabFold environment fix

Two package-version problems were found in the ColabFold environment:

```text
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-colabfold-bioconda
```

Initial versions:

```text
colabfold = 1.5.5
jax       = 0.4.25
jaxlib    = 0.4.25.dev20250110
dm-haiku  = 0.0.16
```

`dm-haiku 0.0.16` failed with:

```text
AttributeError: module 'jax.extend.core' has no attribute 'JaxprEqn'
```

`dm-haiku 0.0.10`, which is the version pinned by ColabFold metadata, then failed with:

```text
AttributeError: module 'jax' has no attribute 'linear_util'
```

The working import combination was:

```text
jax       = 0.4.25
jaxlib    = 0.4.25
dm-haiku  = 0.0.12
```

Applied fixes:

```bash
pip install --no-deps --force-reinstall dm-haiku==0.0.12
pip install --no-deps --force-reinstall jaxlib==0.4.25
```

### ColabFold smoke prediction

Command:

```bash
colabfold_batch \
  --data output/evaluate-mechanical-predictor/colabfold-local-data \
  --msa-mode single_sequence \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-recycle 1 \
  --num-relax 0 \
  output/evaluate-mechanical-predictor/pdb-colab/input-fasta \
  output/evaluate-mechanical-predictor/pdb-colab/results
```

The machine did not expose a usable NVIDIA driver:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

Therefore ColabFold ran on CPU. One of three short smoke sequences completed:

```text
6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm
length = 79 aa
pLDDT  = 58.6
pTM    = 0.387
```

Generated PDB:

```text
output/evaluate-mechanical-predictor/pdb-colab/results/6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
```

The other two smoke sequences failed on the CPU XLA backend with:

```text
Invalid LLVM IR before optimizations:
Floating-point arithmetic operators only work with floating-point types!
%multiply = fmul i16 ...
```

This appears to be a CPU backend compilation limitation or bug in the current ColabFold/JAX stack. It should be retested on a machine/session where CUDA is visible.

### Terminal reward evaluation result

The notebook was rerun after ColabFold prediction and successfully indexed one predicted PDB.

Updated outputs:

```text
output/evaluate-mechanical-predictor/tables/colabfold_predicted_pdb_index.csv
output/evaluate-mechanical-predictor/tables/predicted_mechanical_properties.csv
output/evaluate-mechanical-predictor/tables/prediction_correlations.csv
output/evaluate-mechanical-predictor/figures/prediction_vs_measured_scatter.png
output/evaluate-mechanical-predictor/figures/predicted_pdb_hydrogen_count.png
output/evaluate-mechanical-predictor/evaluation_summary.json
```

Prediction for the one completed structure:

```text
measured toughness          = 0.148
measured tensile_strength   = 1.3
predicted toughness         = 265.404771
predicted strength          = 525.743655
terminal reward             = -0.780269
mean pLDDT                  = 58.965383
hbond_count                 = 0
hydrogen_atom_count         = 0
```

Correlation output:

```text
sequence toughness Pearson/Spearman = NaN
sequence strength  Pearson/Spearman = NaN
idv_mean toughness Pearson/Spearman = NaN
idv_mean strength  Pearson/Spearman = NaN
```

The correlations are undefined because only one predicted PDB was available.

### Interpretation

The sequence-to-structure-to-reward code path is now functionally validated for one sequence:

```text
FASTA -> ColabFold PDB -> hbond/topology features -> random-forest mechanical-property prediction -> terminal reward
```

However, the biological interpretation of this first predicted-structure result is weak for two reasons:

1. The predicted structure has low-to-moderate confidence (`pLDDT ~ 59`, `pTM ~ 0.39`).
2. The ColabFold PDB contains no explicit hydrogens, so the current hbond features collapse to zero and the hbond/topology predictor is operating outside its ideal structural input regime.

Before using this path as a robust external benchmark, the next step should be:

- run ColabFold with visible CUDA rather than CPU;
- generate more than three predicted structures, preferably a controlled subset first;
- add hydrogens or protonate predicted PDBs before hbond-feature extraction;
- recompute Pearson/Spearman once at least three, and preferably dozens of, predicted structures are available.

## 2026-07-16: ColabFold 1.6.2 no-MSA smoke evaluation

### Updated executable paths

The evaluation was rerun with the user-provided ColabFold environment and CUDA toolkit:

```text
ColabFold env: /mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold
CUDA path:     /mnt/data2/jianquanzhao/cuda/cuda-12.8
```

The notebook was updated so the ColabFold executable points to:

```text
/mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold/bin/colabfold_batch
```

The local AlphaFold parameter bridge is still:

```text
output/evaluate-mechanical-predictor/colabfold-local-data/params
```

with symlinks to:

```text
/mnt/nas/jianquanzhao/afdb/params
```

### GPU/CUDA status

The CUDA toolkit exists, but the current runtime session does not expose an NVIDIA driver/device:

```text
nvidia-smi: failed because it couldn't communicate with the NVIDIA driver
/dev/nvidia*: no such file or directory
```

Therefore JAX/ColabFold still ran on CPU:

```text
ColabFold warning: no GPU detected, will be using CPU
```

Using the correct CUDA library directory is important:

```text
/mnt/data2/jianquanzhao/cuda/cuda-12.8/targets/x86_64-linux/lib
```

but this only fixes library discovery. It cannot create a GPU device when the NVIDIA driver is not visible.

### ColabFold command

The three shortest matched spider-silk sequences were predicted in no-MSA mode:

```bash
colabfold_batch \
  --data output/evaluate-mechanical-predictor/colabfold-local-data \
  --msa-mode single_sequence \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-recycle 1 \
  --num-relax 0 \
  --use-pallas false \
  --compile-mode fast \
  output/evaluate-mechanical-predictor/pdb-colab/input-fasta \
  output/evaluate-mechanical-predictor/pdb-colab/results
```

ColabFold 1.6.2 successfully completed all three smoke sequences on CPU:

```text
6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm
  length = 79 aa
  pLDDT  = 58.6
  pTM    = 0.387

6778_3900_Araneidae_Eustala_anastera_AgSp1_nterm
  length = 94 aa
  pLDDT  = 44.8
  pTM    = 0.317

6370_4109_Tetragnathidae_Metleucauge_kompirensis_CySp_nterm
  length = 100 aa
  pLDDT  = 46.5
  pTM    = 0.336
```

### Terminal reward predictions

The notebook was rerun after structure prediction:

```text
model/reward_module/terminal_reward/evaluate_based_predicted.ipynb
```

Updated table outputs:

```text
output/evaluate-mechanical-predictor/tables/colabfold_predicted_pdb_index.csv
output/evaluate-mechanical-predictor/tables/predicted_mechanical_properties.csv
output/evaluate-mechanical-predictor/tables/prediction_correlations.csv
```

Updated figure outputs:

```text
output/evaluate-mechanical-predictor/figures/prediction_vs_measured_scatter.png
output/evaluate-mechanical-predictor/figures/predicted_pdb_hydrogen_count.png
```

Prediction summary:

```text
sequence_length  measured_toughness  measured_strength  predicted_toughness  predicted_strength  mean_plddt  hbond_count  hydrogen_atom_count
100              0.078               0.94               356.135498           837.116284          46.495722   0            0
94               0.106               0.91               354.890629           817.820017          44.965401   0            0
79               0.148               1.30               265.404771           525.743655          58.965383   0            0
```

Correlation results:

```text
level     target     n  pearson    pearson_p  spearman  spearman_p
sequence  toughness  3 -0.922351   0.252531   -1.0      0.000000
sequence  strength   3 -0.992260   0.079259   -0.5      0.666667
idv_mean  toughness  3 -0.922351   0.252531   -1.0      0.000000
idv_mean  strength   3 -0.992260   0.079259   -0.5      0.666667
```

### Analysis

The pipeline is now technically validated:

```text
measured-property CSV + FASTA -> no-MSA ColabFold structure -> hbond/topology feature extraction -> random-forest mechanical-property predictor -> Pearson/Spearman summary
```

However, the current biological/statistical result is not yet reliable:

1. `n=3` is only a smoke-test sample size. Pearson and Spearman values are highly unstable and should not be interpreted as a real external benchmark.
2. no-MSA ColabFold produced low-to-moderate confidence structures for these N-terminal spider-silk fragments (`pLDDT ~ 45-59`, `pTM ~ 0.32-0.39`).
3. ColabFold PDB files do not contain explicit hydrogens, so all hydrogen-bond features collapsed to zero.
4. Because hbond features collapsed to zero, the random forest is mainly responding to residual topology/length-like information, which is not the intended structural-mechanics signal.
5. The negative correlations in this small smoke set should be interpreted as a warning that direct no-MSA predicted structures are not yet a trustworthy input for this hbond-based reward model.

### Conclusion

The software path is runnable, but the current predicted-structure evaluation does not yet support the claim that no-MSA ColabFold structures can directly replace high-quality full-atom structures for the hbond/topology mechanical-property predictor.

The next recommended evaluation should:

- run a larger controlled subset, not just three sequences;
- use a GPU-visible session for scaling;
- add hydrogens/protonate predicted structures before feature extraction;
- filter or weight structures by pLDDT/pTM;
- compare three inputs side by side: original experimental/simulation structures, PyRosetta-relaxed mutant structures, and ColabFold-predicted structures.

## 2026-07-16: Amber-relaxed ColabFold predicted-structure evaluation

### Motivation

The previous no-MSA ColabFold smoke test generated valid predicted PDB files, but the unrelaxed PDBs did not contain explicit hydrogen atoms. Because the hbond/topology mechanical-property predictor was trained on explicit-hydrogen structural features, this caused the hydrogen-bond feature channel to collapse to zero.

This run explicitly enabled Amber/OpenMM relaxation:

```bash
CUDA_PATH='/mnt/data2/jianquanzhao/cuda/cuda-12.8'
export PATH=${CUDA_PATH}/bin:$PATH

colabfold_batch \
  --data output/evaluate-mechanical-predictor/colabfold-local-data \
  --msa-mode single_sequence \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-recycle 1 \
  --amber \
  --num-relax 1 \
  --use-pallas false \
  --compile-mode fast \
  --overwrite-existing-results \
  output/evaluate-mechanical-predictor/pdb-colab/input-fasta \
  output/evaluate-mechanical-predictor/pdb-colab/results
```

OpenMM was available in the ColabFold environment:

```text
openmm = 8.5.2.dev
```

### Runtime status

The ColabFold executable was:

```text
/mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold/bin/colabfold_batch
```

The current session still did not expose a usable NVIDIA driver/device, so ColabFold ran on CPU:

```text
WARNING: no GPU detected, will be using CPU
```

Despite CPU execution, all three short no-MSA smoke sequences completed with Amber relaxation:

```text
6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm
  length = 79 aa
  pLDDT  = 58.8
  pTM    = 0.388
  relaxation time = 10.0 s

6778_3900_Araneidae_Eustala_anastera_AgSp1_nterm
  length = 94 aa
  pLDDT  = 44.8
  pTM    = 0.317
  relaxation time = 8.7 s

6370_4109_Tetragnathidae_Metleucauge_kompirensis_CySp_nterm
  length = 100 aa
  pLDDT  = 46.5
  pTM    = 0.336
  relaxation time = 9.7 s
```

Generated relaxed PDB files:

```text
output/evaluate-mechanical-predictor/pdb-colab/results/6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
output/evaluate-mechanical-predictor/pdb-colab/results/6778_3900_Araneidae_Eustala_anastera_AgSp1_nterm_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
output/evaluate-mechanical-predictor/pdb-colab/results/6370_4109_Tetragnathidae_Metleucauge_kompirensis_CySp_nterm_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
```

### Notebook update

Updated:

```text
model/reward_module/terminal_reward/evaluate_based_predicted.ipynb
```

Changes:

- ColabFold executable points to `/mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold/bin/colabfold_batch`;
- `CUDA_HOME` is recorded as `/mnt/data2/jianquanzhao/cuda/cuda-12.8`;
- the ColabFold command template now includes `--amber --num-relax 1`;
- the notebook execution environment sets `CUDA_PATH` and prepends `${CUDA_HOME}/bin` to `PATH`;
- the summary notes now state that Amber-relaxed ColabFold PDBs are preferred because the hbond predictor expects explicit hydrogens.

### Terminal reward results

The notebook was rerun after Amber relaxation. It correctly selected the relaxed PDB files and regenerated:

```text
output/evaluate-mechanical-predictor/tables/colabfold_predicted_pdb_index.csv
output/evaluate-mechanical-predictor/tables/predicted_mechanical_properties.csv
output/evaluate-mechanical-predictor/tables/prediction_correlations.csv
output/evaluate-mechanical-predictor/figures/prediction_vs_measured_scatter.png
output/evaluate-mechanical-predictor/figures/predicted_pdb_hydrogen_count.png
output/evaluate-mechanical-predictor/evaluation_summary.json
```

Prediction summary:

```text
job_id                                                        length  measured_toughness  measured_strength  predicted_toughness  predicted_strength  mean_plddt  hbond_count  hydrogen_atom_count  hbond_per_residue
6370_4109_Tetragnathidae_Metleucauge_kompirensis_CySp_nterm   100    0.078               0.94               324.202500           705.654877          46.464791   79           708                  0.790000
6778_3900_Araneidae_Eustala_anastera_AgSp1_nterm               94     0.106               0.91               324.828812           670.558291          44.806607   73           713                  0.776596
6999_3943_Tetragnathidae_Tetragnatha_sp_OTU0935_AgSp1_nterm    79     0.148               1.30               282.326461           492.175526          58.974725   67           626                  0.848101
```

Correlation results:

```text
level     target     n  pearson    pearson_p  spearman  spearman_p
sequence  toughness  3 -0.912479   0.268331   -0.5      0.666667
sequence  strength   3 -0.975222   0.142013   -0.5      0.666667
idv_mean  toughness  3 -0.912479   0.268331   -0.5      0.666667
idv_mean  strength   3 -0.975222   0.142013   -0.5      0.666667
```

### Analysis

Amber relaxation solved the most important structural-input problem from the previous run:

```text
unrelaxed ColabFold PDB: hydrogen_atom_count = 0, hbond_count = 0
Amber-relaxed PDB:       hydrogen_atom_count = 626-713, hbond_count = 67-79
```

This means the structure-to-feature part of the predictor is now functioning as intended for predicted structures.

However, the predictive result is still not good in this tiny smoke set:

- the three no-MSA structures have low-to-moderate confidence (`pLDDT ~ 45-59`, `pTM ~ 0.32-0.39`);
- the sample size is only `n=3`, so Pearson/Spearman values are not statistically reliable;
- the predicted toughness and strength are on a much larger numerical scale than the measured spider-silk table values, suggesting cross-dataset target-unit mismatch or domain shift;
- the negative correlations indicate that this tiny no-MSA predicted-structure subset should not be treated as evidence of valid external performance.

### Conclusion

The complete technical pipeline now runs:

```text
CSV + FASTA -> ColabFold no-MSA prediction -> Amber relaxation/add hydrogens -> hbond/topology features -> random-forest mechanical-property prediction -> Pearson/Spearman analysis
```

The main remaining issue is no longer code execution, but scientific validity. For this external spider-silk benchmark, a reliable analysis needs a larger subset, confidence filtering or weighting by pLDDT/pTM, and careful target-unit alignment between the predictor training labels and the measured spider-silk mechanical-property table.

## 2026-07-16: CATH random-test comparison notebook and GPU-gated prediction setup

### Goal

Evaluate the sequence-to-predicted-structure-to-mechanical-property path on a more appropriate dataset: the random test split used by the hbond/topology random-forest mechanical-property predictor.

The intended comparison is:

```text
same CATH/MPPRD random-test entries
  1. ground-truth structure -> hbond/topology features -> random forest prediction
  2. sequence -> ColabFold no-MSA + Amber predicted structure -> hbond/topology features -> random forest prediction
```

### Code

Created:

```text
model/reward_module/terminal_reward/compared_based_predicted.ipynb
```

The notebook implements:

- loading `dataset_random_test_pdb_ids.txt`;
- extracting test entries from `hbond_length_disentanglement_table.csv`;
- writing ColabFold FASTA files;
- GPU-gated ColabFold prediction;
- ground-truth structure evaluation;
- predicted-structure evaluation;
- metric calculation with R2, MAE, RMSE, Pearson, and Spearman;
- paired comparison between ground-truth-feature predictions and predicted-structure predictions.

### Input data

Simulation label table:

```text
/mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv
```

Ground-truth structure directory:

```text
/home/jianquanzhao/data/tsinghua/mpprd/pdbs/pdbs/
```

Random-test split:

```text
outputs/mechanical_property_predictor/hbond_analysis/dataset_random_test_pdb_ids.txt
```

### Test-set extraction result

Prepared:

```text
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/random_test_entries.csv
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/pre_prediction_dataset_summary.json
output/evaluate-mechanical-predictor/tables/cath_random_test_entries.csv
output/evaluate-mechanical-predictor/figures/cath_test_dataset_overview.png
```

Summary:

```text
random_test_ids:        705
selected_test_entries:  705
groundtruth_pdb_exists: 705
FASTA files written:    705
sequence_length_min:    29
sequence_length_median: 87
sequence_length_mean:   108.61
sequence_length_max:    525
```

FASTA files were written to:

```text
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/input-fasta/
```

Predicted structures are expected under:

```text
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/results/
```

### GPU-gated prediction status

The requested prediction environment is:

```text
/mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold
```

with CUDA path:

```text
/mnt/data2/jianquanzhao/cuda/cuda-12.8
```

Before prediction, the notebook configures:

```bash
CUDA_PATH='/mnt/data2/jianquanzhao/cuda/cuda-12.8'
export PATH=${CUDA_PATH}/bin:$PATH
```

The current session could not detect a GPU:

```text
nvidia-smi failed because it couldn't communicate with the NVIDIA driver
```

Following the explicit instruction:

```text
predict structure on gpu: can't detect gpu then terminate
```

ColabFold prediction was not launched in this session. The notebook contains a hard GPU check before prediction, so it will also terminate automatically if run in a non-GPU session.

### Ground-truth baseline

Even though predicted-structure generation was stopped, the ground-truth/precomputed-feature baseline was evaluated using:

```text
/home/jianquanzhao/anaconda24/envs/mprl-vgpt
```

Output:

```text
output/evaluate-mechanical-predictor/tables/cath_test_groundtruth_precomputed_predictions.csv
output/evaluate-mechanical-predictor/tables/cath_test_groundtruth_precomputed_metrics.csv
```

Metrics:

```text
source                         target          n    R2        MAE       RMSE       Pearson   Spearman
groundtruth_precomputed        toughness_v127  705  0.657354  38.6386   63.3918    0.8113    0.8891
groundtruth_precomputed        strength_v128   705  0.306388  65.2115   180.1793   0.5659    0.8326
```

These values reproduce the random-test performance of the hbond random-forest predictor, confirming that the selected test split and model artifact are aligned.

### Current conclusion

The CATH random-test comparison workflow is prepared, and the ground-truth baseline is valid. The predicted-structure branch has not yet been evaluated because this runtime session does not expose a GPU. The next run should be performed in a GPU-visible session; then the notebook can launch ColabFold no-MSA + Amber prediction for the 705 FASTA files and complete the predicted-vs-groundtruth comparison.

## 2026-07-16: runtime GPU detection instead of preflight GPU utility check

### User correction

The GPU gate for ColabFold prediction should not rely on an external preflight GPU utility check. Instead, the workflow should launch ColabFold directly and detect GPU availability from the running ColabFold/JAX log.

### Notebook update

Updated:

```text
model/reward_module/terminal_reward/compared_based_predicted.ipynb
```

Changes:

- removed the preflight GPU utility check from the notebook;
- removed the old `check_gpu_or_raise` function;
- ColabFold is now launched directly through `subprocess.Popen`;
- stdout and stderr are merged and monitored in real time;
- if runtime logs contain a no-GPU/CUDA-backend failure signal, the ColabFold process is terminated immediately;
- `PYTHONUNBUFFERED=1` is set for the ColabFold subprocess;
- non-blocking `select.select` polling is used so the notebook cannot hang forever while waiting for a log line;
- `STARTUP_LOG_TIMEOUT_SECONDS = 180` was added as a safety timeout if no runtime log appears.

Runtime no-GPU patterns currently monitored:

```text
no gpu detected
unable to initialize backend 'cuda'
unable to initialize backend cuda
failed to initialize cuda
cuda_error_no_device
could not find cuda
```

Runtime status files:

```text
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/colabfold_runtime.log
output/evaluate-mechanical-predictor/pdb-colab/CATH-test/colabfold_runtime_status.json
```

### Validation

A short validation run was performed by launching the actual ColabFold command on the CATH-test FASTA directory. The process was not allowed to continue on CPU.

Detected runtime log:

```text
Running colabfold 1.6.2
WARNING: no GPU detected, will be using CPU
```

Validation status:

```text
terminated_due_to_no_gpu: true
terminated_due_to_log_timeout: false
return_code: -15
```

This confirms the new behavior:

```text
launch ColabFold directly -> inspect runtime log -> terminate immediately if ColabFold reports CPU/no-GPU execution
```

## 2026-07-16: MechanoPro-DB external evaluation notebook

### Goal

Prepare a third-party experimental benchmark for the hbond/topology mechanical-property predictor.

The external target is:

```text
Highest unfolding forces/ Clamp forces [pN]
```

This is an experimental unfolding/clamp force, not the same target as the simulated training labels `v127` and `v128`. Therefore the notebook evaluates association/correlation between the experimental force and both predictor heads:

```text
predicted toughness head: v127
predicted strength head:  v128
```

### Code

Created:

```text
model/reward_module/mechanical-properties-predictor/compared_based_predicted-MechanoPro-DB.ipynb
```

The notebook was adapted from:

```text
model/reward_module/terminal_reward/compared_based_predicted.ipynb
```

It implements:

- MechanoPro-DB CSV parsing;
- force-label parsing;
- duplicate PDB ID aggregation by mean experimental force;
- RCSB PDB download by PDB ID;
- PDB-derived sequence parsing;
- ColabFold no-MSA + Amber prediction setup;
- runtime GPU detection from ColabFold logs;
- predicted-structure evaluation;
- groundtruth-structure evaluation;
- correlation analysis between experimental force and both predicted heads.

### Input

CSV:

```text
model/reward_module/mechanical-properties-predictor/MechanoPro-DB.csv
```

Important columns:

```text
Name (Click for details)
PDB ID
Highest unfolding forces/ Clamp forces [pN]
```

The raw CSV contains a leading space in the force column header, so the notebook strips column names before parsing.

### Duplicate handling

The CSV contains repeated PDB IDs, often due to different experimental pulling modes or constructs. The notebook aggregates by PDB ID:

```text
experimental_force_pn = mean(force values for the same PDB ID)
```

It also retains:

```text
n_measurements
force_std_pn
force_min_pn
force_max_pn
entry_descriptions
techniques
pulling_modes
```

### Prepared data

Parsed and prepared:

```text
raw rows:                  85
clean measurement rows:    85
unique PDB entries:        38
RCSB PDB downloads:        38 / 38
entries with PDB sequence: 38 / 38
FASTA files:               38
sequence length range:     53 to 950 aa
sequence length median:    249.5 aa
```

Generated files:

```text
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_raw_normalized.csv
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_clean_measurements.csv
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_pdb_aggregated_entries.csv
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_entries_with_sequences.csv
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_colabfold_input_sequences.csv
```

Downloaded RCSB PDB files:

```text
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/groundtruth-rcsb-pdb/
```

Prepared ColabFold FASTA files:

```text
output/evaluate-mechanical-predictor/pdb-colab/compared_based_predicted-MechanoPro-DB/input-fasta/
```

Predicted structures are expected under:

```text
output/evaluate-mechanical-predictor/pdb-colab/compared_based_predicted-MechanoPro-DB/results/
```

### Figures

Generated:

```text
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/figures/mechanopro_force_and_duplicates.png
output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/figures/mechanopro_sequence_length_vs_force.png
```

### ColabFold configuration

The notebook uses:

```text
conda env: /mnt/data1/home/jianquanzhao/anaconda24/envs/colabfold
CUDA path: /mnt/data2/jianquanzhao/cuda/cuda-12.8
```

Before prediction it sets:

```bash
CUDA_PATH='/mnt/data2/jianquanzhao/cuda/cuda-12.8'
export PATH=${CUDA_PATH}/bin:$PATH
```

Per the user instruction, the notebook does not configure the dynamic-library search path. This avoids the GPU-detection issue observed when that path is manually overridden.

ColabFold command template:

```bash
colabfold_batch \
  --data output/evaluate-mechanical-predictor/colabfold-local-data \
  --msa-mode single_sequence \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-recycle 1 \
  --amber \
  --num-relax 1 \
  --use-pallas false \
  --compile-mode fast \
  output/evaluate-mechanical-predictor/pdb-colab/compared_based_predicted-MechanoPro-DB/input-fasta \
  output/evaluate-mechanical-predictor/pdb-colab/compared_based_predicted-MechanoPro-DB/results
```

### Current execution status

The data preparation and RCSB download stages were completed successfully.

ColabFold prediction was not executed in this turn. The notebook contains the prediction logic and should be run in the GPU-visible ColabFold environment.

A direct batch groundtruth-structure evaluation was attempted in the interactive session, but it was interrupted because some larger RCSB structures make direct geometry-based hbond extraction slow. The notebook still contains the full evaluation logic; for production evaluation, it should be run as a notebook/job, or the groundtruth evaluation should be batched.

### Interpretation note

This benchmark is scientifically useful but not target-identical:

- training labels are simulated `v127` and `v128`;
- MechanoPro-DB provides experimental unfolding/clamp force;
- duplicate PDB IDs can reflect different pulling modes, so mean aggregation is a pragmatic first baseline but not the only possible treatment.

The first analysis should therefore emphasize Pearson/Spearman trends rather than direct R2/absolute-error interpretation.

## 2026-07-17 MechanoPro-DB Failure Analysis and Top-k Screening Check

### Context

The MechanoPro-DB external benchmark was used to test whether the current structure-based mechanical-property predictor can transfer from the simulated CATH/MPPRD-style labels to experimental unfolding/clamp force.

Current correlation summary on 38 aggregated PDB entries:

| Structure source | Predictor | Pearson | Spearman | Interpretation |
| --- | ---: | ---: | ---: | --- |
| RCSB groundtruth | toughness v127 | -0.216 | -0.272 | weak negative association |
| RCSB groundtruth | strength v128 | 0.064 | -0.030 | almost no association |
| RCSB groundtruth | terminal reward | -0.075 | -0.203 | weak negative association |
| ColabFold predicted | toughness v127 | -0.186 | -0.374 | negative ranking association |
| ColabFold predicted | strength v128 | -0.171 | -0.392 | negative ranking association |
| ColabFold predicted | terminal reward | -0.198 | -0.387 | negative ranking association |

### Why the External Dataset Performs Poorly

1. The training targets and external endpoint are not identical. The random-forest predictor was trained on simulated `v127` and `v128`, while MechanoPro-DB reports experimental unfolding/clamp force in pN.
2. Experimental force is strongly loading-path dependent. Pulling residues, clamp geometry, construct boundary, mutation state, oligomeric state, and assay protocol can change the measured force even for the same fold.
3. The current features describe static topology and hydrogen-bond networks, but do not encode the pulling vector, force-propagation path, transition state, or unfolding pathway.
4. Duplicate PDB IDs in MechanoPro-DB can correspond to different measurement modes. Averaging duplicates is a reasonable first baseline, but it can erase real protocol-specific mechanical signals.
5. RCSB structures and ColabFold structures are equilibrium structures. They are not force-loaded conformational trajectories.
6. Hydrogen handling, missing atoms, chain selection, biological assembly, and protonation differ across PDB and predicted structures; these directly affect hydrogen-bond features.
7. The external benchmark is small after PDB-ID aggregation (`n=38`), so outliers and protocol heterogeneity can dominate correlation metrics.

### Top-k Accuracy / Enrichment Analysis

Definition: a hit means overlap between the model top-k by predicted score and the experimental-force top-k by pN. For `n=38`, random top-5 baseline hit rate is `5/38 = 0.132`.

Top-5 results:

| Structure source | Score | Hit count | Hit rate | Enrichment | Predicted top-5 mean force |
| --- | ---: | ---: | ---: | ---: | ---: |
| RCSB groundtruth | toughness v127 | 0/5 | 0.000 | 0.00 | 58.45 pN |
| RCSB groundtruth | strength v128 | 0/5 | 0.000 | 0.00 | 129.80 pN |
| RCSB groundtruth | terminal reward | 0/5 | 0.000 | 0.00 | 81.25 pN |
| ColabFold predicted | toughness v127 | 1/5 | 0.200 | 1.52 | 353.25 pN |
| ColabFold predicted | strength v128 | 0/5 | 0.000 | 0.00 | 99.80 pN |
| ColabFold predicted | terminal reward | 0/5 | 0.000 | 0.00 | 95.85 pN |

Diagnostic inverse-toughness ranking:

| Structure source | Diagnostic score | Hit count | Hit rate | Enrichment | Hit ID |
| --- | ---: | ---: | ---: | ---: | --- |
| RCSB groundtruth | inverse toughness v127 | 1/5 | 0.200 | 1.52 | 1UBQ |
| ColabFold predicted | inverse toughness v127 | 1/5 | 0.200 | 1.52 | 1UBQ |

The inverse-ranking diagnostic is not a proposed reward. It shows that the negative correlation has practical ranking consequences: the current simulated-label predictor is not aligned with experimental unfolding/clamp force in MechanoPro-DB.

### Files Updated / Generated

- Appended failure-analysis and top-k evaluation cells to `model/reward_module/mechanical-properties-predictor/compared_based_predicted-MechanoPro-DB.ipynb`.
- Generated `output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_topk_hit_analysis.csv`.
- Generated `output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_top10_leaderboard_by_score.csv`.
- Generated `output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/tables/mechanopro_topk_summary.json`.
- Generated `output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/figures/mechanopro_topk_hit_rate.png`.
- Generated `output/evaluate-mechanical-predictor/compared_based_predicted-MechanoPro-DB/figures/mechanopro_topk_enrichment.png`.

### Conclusion

The current predictor is useful inside the simulated-label domain, but it should not yet be treated as a general experimental unfolding-force predictor. For reward design, it is safer to use it as a structure/topology prior or auxiliary reward, while building an additional calibration layer for experimental force endpoints that includes pulling geometry, construct definition, and protocol information.

## 2026-07-20 Spider Silkom Material-Property Evaluation

### Goal

Evaluate whether the current structure-based SMD mechanical-property predictor can explain spider silk material-level properties, and use the result to decide how this predictor should guide future mechanical-protein screening.

Input dataset:

```text
/mnt/nas/jianquanzhao/data/mprl/spider_silkom_database
```

Key files:

```text
infor_csv/compressed_species_silk_db.csv
infor_csv/silk_sequence_performance_db.csv
colab-pdb/*_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
```

### Method

The spider silkom dataset is a harder external benchmark than the SMD test set because the labels are material-level properties of mixed silk produced by each spider taxon. A single row in `compressed_species_silk_db.csv` is indexed by `taxon_key` and can include multiple silk protein types in `silk_types_included`.

Evaluation strategy:

1. Parse `compressed_species_silk_db.csv` as taxon-level material labels.
2. Parse `silk_sequence_performance_db.csv` as taxon-level silk protein sequence/structure entries.
3. Match each FASTA file to its ColabFold relaxed PDB in `colab-pdb`.
4. Predict SMD-like toughness, strength, and terminal reward for every unique relaxed PDB.
5. Aggregate protein-level predictions to taxon level using two schemes:
   - `structure_weighted`: every unique predicted structure contributes equally.
   - `type_balanced`: average structures within each `taxon_key + silk_type`, then average silk types equally. This is the primary interpretation because mixed silk material properties are not simply proportional to the number of database fragments.
6. Compare taxon-level predicted scores with material `toughness` and `tensile_strength` by Pearson, Spearman, and top-k enrichment.

### Engineering Note

Full evaluation requires thousands of structure-level hydrogen-bond feature extractions. The original hydrogen-bond search was too slow for this scale, so `model/reward_module/terminal_reward/features.py` was accelerated with a spatial-index implementation:

- donor-H pairs are restricted to the same residue before distance checking;
- H-acceptor candidate atoms are queried with `scipy.spatial.cKDTree`;
- all original hydrogen-bond thresholds and feature definitions are unchanged;
- a fallback loop implementation remains available if SciPy is unavailable.

On the spider silkom structures, this reduced a typical structure evaluation from about 1-3 seconds to about 0.2-0.3 seconds. With 16 workers, the full cache was generated at about 55 structures/second.

### Data Coverage

| Item | Count |
| --- | ---: |
| compressed taxon rows | 219 |
| taxa with material labels | 204 |
| sequence rows | 12,228 |
| unique taxon/silk/structure pairs | 5,446 |
| unique predicted structures evaluated successfully | 5,310 |
| taxa with any structure prediction | 218 |
| taxa with both material label and prediction | 203 |

### Correlation Results

Primary type-balanced taxon-level results:

| Comparison | n | Pearson | Spearman | Spearman p |
| --- | ---: | ---: | ---: | ---: |
| material toughness vs predicted SMD toughness | 203 | -0.085 | -0.138 | 0.0499 |
| material toughness vs terminal reward | 203 | -0.012 | -0.065 | 0.353 |
| material tensile strength vs predicted SMD strength | 203 | -0.036 | -0.073 | 0.302 |
| material tensile strength vs terminal reward | 203 | -0.056 | -0.112 | 0.113 |

Structure-weighted aggregation was also weak:

| Comparison | n | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| material toughness vs predicted SMD toughness | 203 | 0.001 | 0.007 |
| material toughness vs predicted SMD strength | 203 | 0.107 | 0.088 |
| material tensile strength vs predicted SMD strength | 203 | -0.030 | -0.053 |
| material tensile strength vs terminal reward | 203 | -0.032 | -0.060 |

### Top-k Screening Results

For type-balanced aggregation, top-10% means `k=21` out of 203 taxa. The random hit-rate baseline is `21/203 = 0.103`.

| Experimental target | Score | Hit count | Hit rate | Enrichment | Lift vs overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| material toughness | predicted SMD toughness | 1/21 | 0.0476 | 0.46 | 0.754 |
| material tensile strength | predicted SMD strength | 2/21 | 0.0952 | 0.92 | 0.910 |
| material toughness | terminal reward | 3/21 | 0.1429 | 1.38 | 0.870 |
| material tensile strength | terminal reward | 2/21 | 0.0952 | 0.92 | 0.895 |

The modest terminal-reward top-10% enrichment for toughness is not enough to support direct material-level screening. The predicted top-k mean material property is still below the global average for most settings, which means the current predictor does not reliably select high-performing material-level silk taxa.

### Files Generated

Notebook:

```text
model/reward_module/mechanical-properties-predictor/evalutaed-spider_silkom.ipynb
```

Tables:

```text
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_structure_predictions.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_sequence_structure_prediction_pairs.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_taxon_aggregated_predictions.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_taxon_silk_type_predictions.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_taxon_type_balanced_predictions.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_correlation_metrics.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_topk_analysis.csv
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/tables/spider_silkom_summary.json
```

Figures:

```text
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/figures/spider_silkom_material_property_distribution.png
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/figures/spider_silkom_predicted_vs_material_scatter.png
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/figures/spider_silkom_spearman_heatmap.png
output/evaluate-mechanical-predictor/evalutaed-spider_silkom/figures/spider_silkom_topk_enrichment.png
```

### Conclusion

The current structure-based predictor should not be used as a direct material-level spider silk property predictor. This result is biologically reasonable: silk material properties are emergent outcomes of multiple proteins, spinning conditions, hierarchical assembly, crystallinity, water content, fiber diameter, and experimental protocol.

Recommended use for future screening:

1. Use the current predictor as a single-protein structural prior or auxiliary reward, not as the final material-level objective.
2. Build a taxon-level or mixture-level model that combines per-protein structure scores with composition features such as `silk_types_included`, amino-acid composition, poly-A content, GPGXX motif counts, fragment count, and total combined length.
3. Prefer type-balanced aggregation when connecting protein-level predictions to mixed-silk material labels.
4. Treat direct sequence/structure-to-material screening as a multi-instance learning problem rather than a single-protein regression problem.

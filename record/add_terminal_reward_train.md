## 2026-07-20 terminal reward training integration

### Goal

Check and complete the integration of the structure-based mechanical-property predictor into the RL terminal reward, keep the base-version terminal reward ratios equal, normalize the step reward to a 0-1 scale, split the CATH PDB data, and run a 32-epoch training smoke test.

### Code Changes

1. `training.py`
   - Added CLI switches:
     - `--step-reward-scale`
     - `--terminal-reward-scale`
     - `--no-terminal-reward`
     - `--terminal-reward-artifact`
     - `--terminal-predicted-pdb-dir`
   - The training entry now builds `EqualWeightDualStructureTerminalRewardCalculator` by default and passes it into `MechanicalProteinEnv`.
   - The terminal reward default uses:
     - strength:toughness = 1:1
     - PyRosetta terminal structure:predicted structure = 1:1 when a matching predicted PDB exists.

2. `model/reward_module/terminal_reward/calculator.py`
   - Changed single-structure objective scalarization from weighted sum to weighted mean, so equal strength/toughness weights keep the reward scale stable.
   - Added `EqualWeightDualStructureTerminalRewardCalculator`.
   - The new wrapper scores the PyRosetta terminal pose and resolves a matching predicted PDB by source PDB stem. If both are present, it returns the average of the two structure rewards.
   - If no predicted structure is available, it falls back to the PyRosetta terminal pose and records `predicted_structure_missing=1.0`.

3. `model/environment_module/environment.py`
   - Episode finalization now calls `terminal_reward_calculator.evaluate_episode(relaxed_pose=..., source_pdb_path=...)` when available, so terminal reward has both the terminal pose and the source PDB identity.
   - PDB preprocessing was strengthened for CATH-style files:
     - always writes a normalized temporary PDB when cleaning is enabled;
     - rewrites ATOM/HETATM records into standard fixed-width PDB lines;
     - infers malformed element columns such as `XP1`;
     - converts non-positive occupancy to `1.00`, which prevents PyRosetta from loading valid coordinate records as an empty pose.

4. `model/reward_module/reward_calculators.py`
   - Step reward is now a bounded weighted mean in 0-1:
     - collision penalty: `1 - clip(collision_loss / collision_scale, 0, 1)`
     - backbone H-bond delta: `0.5 + 0.5 * tanh(delta / scale)`
     - sidechain H-bond delta: `0.5 + 0.5 * tanh(delta / scale)`
     - local RMSD penalty: `1 - clip(local_rmsd / local_rmsd_scale, 0, 1)`
   - This avoids raw Rosetta collision/energy terms dominating the step reward.

5. `tests/test_reward_calculators.py`
   - Added unit tests for the bounded step-reward mapping helpers.

### Data Split and Output Directories

- Full CATH input path: `/mnt/nas/jianquanzhao/data/mprl/pdbs/cath/`
- Full split files:
  - train: `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt`, 7701 PDBs
  - validation: `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt`, 856 PDBs
- Training output root:
  - `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/`
- Figures copied to:
  - `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/figures/`

### Training Test

The ESM2 mixed-length path was attempted first, but `fair-esm` tried to download `esm2_t33_650M_UR50D.pt` into `/tmp/mprl-vgpt-cache/torch/hub/checkpoints/`. To avoid a long model download during this integration test, the completed 32-epoch smoke run used a fixed PyRosetta pose-length subset with the default one-hot observation encoder.

Completed run:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/condabin/conda run --no-capture-output -n mprl-vgpt python training.py \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_predicted_pyrosetta_length_smoke.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/terminal_reward_pyrosetta_length_smoke_32 \
  --mode single --device auto --epochs 32 --episodes-per-epoch 1 \
  --train-batch-size 1 --max-steps 1 --no-minimize --continue-on-update-error \
  --replay-warmup-size 4 --micro-batch-size 2 --gradient-accumulation-steps 1 \
  --replay-capacity 256 --checkpoint-every 16 --plot-every-episodes 4 \
  --validate-every 0 --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps 1 --no-resume-logs
```

Run result:

- episodes: 32
- environment steps: 32
- optimization steps: 29
- predicted structure matches: 32/32
- checkpoints:
  - `checkpoints/agent.pt`
  - `checkpoints/agent_final.pt`
  - `checkpoints/replay_buffer.npz`
  - `checkpoints/replay_buffer_final.npz`
- generated plots:
  - `episode_reward.png`
  - `episode_length.png`
  - `epsilon.png`
  - `optimization_loss.png`
  - `td_error.png`
  - `grad_norm.png`
  - `q_values.png`
  - `step_reward.png`
  - `reward_components.png`
  - `terminal_reward.png`

### Training Metrics Summary

- step reward:
  - min: 0.4171
  - mean: 0.5188
  - max: 0.7841
  - conclusion: step reward is now bounded and no longer numerically dominated by the raw energy/collision term.
- terminal reward:
  - min: -0.2093
  - mean: -0.0227
  - max: 0.2688
  - conclusion: terminal reward has a smaller but visible contribution to the final reward.
- episode total reward:
  - min: 0.2275
  - mean: 0.4961
  - max: 0.7781
- optimization:
  - loss decreased from 0.2343 to 0.0889 during the smoke run.
  - mean absolute TD error decreased from 0.6662 to 0.3425.
  - grad norm stayed finite and below the 10.0 clipping threshold.

### Notes and Next Steps

1. The completed 32-epoch smoke test validates the reward integration and training loop, but it is intentionally not a full mixed-length production run.
2. For full mixed-length training, use `--observation-encoder esm2` after placing the ESM2 checkpoint in the torch hub cache or configuring an available cache path. That path enables padded replay and per-residue Q output.
3. The terminal reward is now usable in the base RL framework. The current base version uses equal structure weights and equal mechanical-objective weights; later experiments can introduce quality-weighted predicted/PyRosetta averaging or tune terminal-vs-step reward scale.

## 2026-07-20 update: ESM2 mixed-length terminal-reward training

### Path Clarification

The clarified training environment path in the task was `/home/jianquanzhao/anaconda24/envs/mprl-vgp`, but this environment does not exist on the current machine. The available training environment is:

```bash
/home/jianquanzhao/anaconda24/envs/mprl-vgpt
```

The ESM2 model files are already available locally at:

```bash
/mnt/data1/home/jianquanzhao/data/models/esm
```

To avoid any model download during training, `training.py` and `model/encoding_module/esm2_encoder.py` were updated with `--esm-model-dir`, so the ESM2 encoder can load `esm2_t33_650M_UR50D.pt` directly from the local model directory.

### Additional Code Changes

1. `model/encoding_module/esm2_encoder.py`
   - Added `model_dir` support.
   - When `model_dir` is supplied, the encoder loads the local fair-esm checkpoint through `esm.pretrained.load_model_and_alphabet_local(...)`.
   - Added PyTorch safe-global registration for `argparse.Namespace`, which is needed by newer PyTorch checkpoint loading behavior.

2. `training.py`
   - Added `--esm-model-dir`.
   - Passed the local ESM model directory into `ESM2SequenceEncoder`.

The previously implemented reward changes remain active:

- step reward is calculated as a normalized weighted mean of 0-1 component scores;
- terminal reward uses the hbond/topology random-forest mechanical-property predictor;
- strength:toughness uses equal 1:1 objective weights;
- PyRosetta terminal structure:sequence-predicted structure uses equal 1:1 structure weights when the predicted PDB exists.

### Mixed-Length Training Command

```bash
/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python training.py \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_predicted_smoke_32.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/terminal_reward_esm_mixed_length_32 \
  --mode single --device auto --epochs 32 --episodes-per-epoch 1 \
  --train-batch-size 1 --max-steps 1 \
  --observation-encoder esm2 --embedding-dim 1280 --esm2-device auto \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error \
  --replay-warmup-size 4 --micro-batch-size 2 --gradient-accumulation-steps 1 \
  --replay-capacity 256 --checkpoint-every 16 --plot-every-episodes 4 \
  --validate-every 0 --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps 1 --no-resume-logs
```

### Data Split

- full train index: `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt`, 7701 PDBs
- full validation index: `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt`, 856 PDBs
- 32-entry mixed-length smoke index with predicted-structure matches: `/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_predicted_smoke_32.txt`

### Training Result

The ESM2 mixed-length training smoke test completed successfully.

- episodes: 32
- environment steps: 32
- optimization steps: 29
- elapsed time: about 72 seconds
- ESM2 device: CUDA
- replay mode: variable length
- predicted structure matches: 32/32
- candidate structure lengths:
  - min: 36 aa
  - mean: 108.66 aa
  - max: 290 aa
  - unique lengths: 18

Output directory:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/terminal_reward_esm_mixed_length_32
```

Figures copied to:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/figures/esm_mixed_length_32
```

### Reward Analysis

The raw `steps.jsonl` field `reward` is the environment-returned total reward for this one-step smoke setting, so it includes both normalized step reward and terminal reward. Therefore it can be outside 0-1. The actual normalized step reward should be read from the bounded component scores.

Bounded step reward reconstructed from normalized components:

- min: 0.3347
- mean: 0.5714
- max: 0.8392

This confirms that the step reward normalization is active and the raw collision/energy term no longer dominates the scale.

Environment total reward, equal to step reward plus terminal reward in this one-step smoke test:

- min: -0.4255
- mean: 0.4524
- max: 1.3728

Terminal reward:

- min: -0.9587
- mean: -0.1191
- max: 0.8742

The terminal reward range is now large enough to affect learning direction, while the step reward remains stable and bounded.

### Optimization Analysis

- loss:
  - first: 0.3375
  - mean: 0.2585
  - last: 0.1341
- mean absolute TD error:
  - first: 0.7201
  - mean: 0.6408
  - last: 0.4531
- gradient norm:
  - min: 0.2883
  - mean: 1.0658
  - max: 1.6471

The smoke run shows finite gradients, decreasing loss, and decreasing TD error. This is enough to validate that the reward integration, mixed-length ESM observation path, padded replay path, and per-residue Q head are connected correctly.

### Conclusion

The structure-based mechanical-property predictor is now part of the RL terminal reward in the base training path. For the base version, all terminal reward ratios are equal:

- strength:toughness = 1:1
- PyRosetta terminal packed structure:predicted structure = 1:1 when the predicted structure exists

The step reward has been normalized to a bounded 0-1 weighted average before the terminal reward is added. The completed 32-epoch ESM2 mixed-length smoke run validates the intended training path on multiple protein lengths.

For the next larger experiment, the main remaining issue is not code connectivity but experimental scale: increase `--max-steps`, use more episodes per epoch, keep validation enabled, and evaluate whether terminal reward should be delayed, scaled, or quality-weighted after the base behavior is stable.

## 2026-07-20 overnight full-data training launch

### Goal

After the 32-episode smoke test passed, an overnight training run was launched to produce a larger first result for tomorrow's analysis.

### Practical Schedule Choice

A mathematically full run of `32 epochs * 7701 train PDBs * 5 max steps` would be too large for an overnight experiment. The launched run keeps 32 epoch-level checkpoints/plots but sets `episodes_per_epoch=240`, giving:

- planned episodes: 7680
- max steps per episode: 3
- planned environment steps: up to 23040

This is close to one full train-split pass in total episode count while still preserving per-epoch trend monitoring.

### Predicted-Structure Coverage

The current predicted-structure directory is:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test
```

It contains 1410 predicted PDB files. Exact matching against the full train index found:

- matched train entries: 635 / 7701
- coverage: 8.2457%

Therefore, in this overnight full-data run:

- entries with a matching predicted PDB use PyRosetta terminal packed structure:predicted structure = 1:1;
- entries without a matching predicted PDB fall back to the PyRosetta terminal packed structure reward and record `predicted_structure_missing=1.0`.

This is acceptable for launching the overnight run, but strict all-episode 1:1 dual-structure training requires predicting structures for the full train split first.

### Launch Command

```bash
RUN_DIR=/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720

setsid env PYTHONUNBUFFERED=1 /home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python training.py \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir "$RUN_DIR" \
  --mode single --device auto \
  --epochs 32 --episodes-per-epoch 240 --train-batch-size 8 --max-steps 3 \
  --observation-encoder esm2 --embedding-dim 1280 --esm2-device auto \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error \
  --replay-warmup-size 512 --micro-batch-size 4 --gradient-accumulation-steps 4 \
  --replay-capacity 50000 --target-sync-interval 500 \
  --checkpoint-every 240 --plot-every-episodes 240 --rolling-window 100 \
  --validate-every 240 --validation-episodes 16 --enable-tensorboard \
  --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps 25 --no-resume-logs --no-save-candidates \
  > "$RUN_DIR/train_stdout.log" 2>&1 < /dev/null &
```

### Runtime State

- PID: `3409529`
- run directory:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720
```

- stdout/stderr log:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720/train_stdout.log
```

- tensorboard directory:

```bash
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720/tensorboard
```

### Startup Check

The run was confirmed alive after startup:

- process state: running
- planned episodes: 7680
- CUDA visible to code: 4 GPUs
- ESM2 checkpoint loaded from the local model directory
- replay mode: variable length
- episode loop reached at least episode 46/7680
- `logs/steps.jsonl`, `logs/episodes.jsonl`, `logs/episodes.csv`, and TensorBoard event files were created and growing

### Monitoring Commands

```bash
tail -f /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720/train_stdout.log
```

```bash
ps -p 3409529 -o pid,ppid,sid,stat,etime,cmd
```

```bash
tensorboard --logdir /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_overnight_20260720/tensorboard
```

### Notes for Tomorrow's Analysis

1. Optimization starts after replay warmup reaches 512 transitions, so the earliest several hundred steps are data collection.
2. Plots and checkpoints are generated every 240 episodes, approximately once per epoch in this schedule.
3. If final results are noisy, the first follow-up should be to predict structures for the full train split so the terminal reward can use true dual-structure 1:1 scoring in every episode.

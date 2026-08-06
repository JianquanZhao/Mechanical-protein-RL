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

## 2026-07-27 analysis of the `max_steps=50` full training run

Run:

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_fully-change-max-steps
```

### Overall conclusion

Reducing `max_steps` from 1024 to 50 caused a real numerical recovery, but the
current run has not reached a healthy policy:

1. Q, loss, TD error, and pre-clipping gradient norm peaked and then declined,
   with no NaN or infinity.
2. Latest mean Q is still about `111.78`, while the maximum observed episode
   return is `39.33` and the current reward-scale reference is about `40`.
   Q-value overestimation remains.
3. Latest pre-clipping gradient norm is about `38.71`; `99.16%` of optimizer
   updates exceeded the clip threshold of 10. The applied gradients were clipped,
   but training remains almost permanently in the clipped regime.
4. Repeated operations remain severe. In the 250k+ low-epsilon phase, `76.24%`
   of sequence changes revisited a changed position, `63.46%` immediately changed
   the same position again, and `47.62%` exactly reversed the previous change.
5. Terminal mechanical reward has no measurable improvement trend after epsilon
   decay: correlation with episode number is `0.0025`.
6. The run stopped during replay-buffer saving at episode 18959. `agent.pt` is
   readable at environment step 948000, but the approximately 9.66 GB
   `replay_buffer.npz` is corrupt and cannot be used to resume.

This is a useful diagnostic run, not a completed or validated optimization model.

### Run status

- Completed episodes: `18,960 / 246,432` (`7.69%`)
- Environment steps: `948,000`
- Optimizer updates: `947,489`
- Training structures seen: all `7,701`, about 2.46 dataset passes
- Every completed episode used exactly 50 steps
- Validation records: `1,264`
- Non-finite optimization metrics: `0`
- No training process is currently running
- The full schedule contains `12,321,600` environment steps

The observed 7.69% took about 5.66 days. At this throughput, the complete
32-epoch schedule would take approximately 73-74 days.

### Numerical trend

| Metric | First 10k mean | Peak 25k-bin mean | Latest 25k-bin mean |
|---|---:|---:|---:|
| Huber loss | 0.349 | 229.670 | 2.587 |
| Absolute TD error | 0.591 | 230.169 | 2.942 |
| Q value | 11.724 | 6116.109 | 111.777 |
| Target Q | 11.582 | 6011.063 | 109.644 |
| Q - target Q | 0.142 | 105.046 | 2.133 |
| Pre-clipping gradient norm | 5.570 | 930.662 | 38.709 |

The decline is genuine, but Q remains above the empirical return scale and above
target Q. In
`model/agent_module/ddqn_agent.py`, `clip_grad_norm_` returns the pre-clipping
norm, so the update was limited to 10; however, `939,561 / 947,489` clipped
updates show that the TD targets and gradients are still poorly scaled.

Likely causes include the always-positive step reward, `gamma=0.99`, replay data
dominated by cycles, and a non-Markov observation. The ESM2 encoder discards the
Pose and encodes sequence only, while reward and transition dynamics depend on
the repacked structure, history, and remaining steps. The same sequence can
therefore receive contradictory Q targets.

### Repeated-action problem

| Policy phase | Revisit | Immediate same position | Two-step reversal | Mean unique changed positions |
|---|---:|---:|---:|---:|
| Epsilon decay, 0-50k | 42.76% | 17.99% | 14.09% | 28.05 |
| Early greedy, 50k-250k | 78.29% | 67.81% | 55.77% | 10.64 |
| Late greedy, 250k+ | 76.24% | 63.46% | 47.62% | 11.64 |

`max_steps=50` limits cycle length but does not remove the cycle. The run has
`prevent_revisit_positions=false`; the current action mask removes only the
current amino-acid no-op and permits all previously edited positions.

Recommended solution:

1. Enable `prevent_revisit_positions` for the next base experiment.
2. Add normalized remaining-step budget to the observation.
3. If revisits are later required, include mutation history/structural state and
   penalize immediate reversals.
4. Use potential-based shaping,
   `r_step = Phi(s_next) - Phi(s_current)`, so a two-step cycle has approximately
   zero shaping return.

### Reward-signal problem

- Mean step reward: `0.6696`
- Mean episode total: `33.388`
- Mean terminal reward: `-0.092`
- Mean absolute terminal reward: `0.432`
- Mean absolute terminal contribution relative to episode total: about `1.29%`
- Post-epsilon total-reward trend correlation: `0.0206`
- Validation trend correlation: `0.0434`

The normalized step terms are highly saturated:

- collision score equals zero in `34.43%` of steps;
- backbone-H-bond score equals neutral 0.5 in `97.27%`;
- sidechain-H-bond score equals neutral 0.5 in `71.68%`;
- local-RMSD score is at least 0.99 in `94.49%`.

The policy can collect an almost constant positive local reward much more easily
than improve terminal mechanical properties. Recommended changes are to center
step shaping around zero, scale cumulative step shaping by episode length, and
log strength/toughness changes relative to the initial structure. Evaluation
must compare against random-policy and wild-type baselines.

### Terminal dual-structure issue

Only `1,566 / 18,960` episodes (`8.26%`) found a predicted PDB. The remaining
`91.74%` used the PyRosetta terminal structure alone.

The current predicted-structure lookup uses `source_pdb_path` and its stem, not a
hash or identifier of the mutated terminal sequence. Therefore, unless a
predicted structure was generated for that exact terminal sequence, this term is
a constant source-sequence prediction and cannot guide action selection. When
averaged 1:1, it can dilute the action-dependent PyRosetta terminal reward.

Recommended solution:

1. Use the PyRosetta terminal Pose for online action-dependent terminal reward.
2. Predict terminal sequences asynchronously/offline and cache by sequence hash.
3. Verify exact sequence identity before using a predicted PDB.
4. Report exact-match coverage before describing the experiment as 1:1
   dual-structure reward.

The random-forest artifact also emits a scikit-learn compatibility warning: it
was trained under 1.6.1 and loaded under 1.7.2. The predictor environment should
match the training version or be regression-tested after re-export.

### Run termination and storage

The confirmed stop point is replay serialization. The variable-length buffer
stores 50,000 ESM state and next-state arrays plus action masks, and writes them
with `np.savez_compressed`. The logger also retains all step and optimization
dictionaries in memory while appending JSONL. stdout reached about 7.17 GB.

OOM or an external kill is plausible but not directly confirmed because kernel
OOM records were not accessible. The corrupt replay archive and missing
`ReplayBuffer save complete` record are confirmed.

Recommended solution:

1. Save `agent.pt` frequently and replay snapshots much less often.
2. Use atomic temporary checkpoint writes followed by validation and rename.
3. Store compact sequence/token inputs or float16/chunked replay data instead of
   duplicated float32 ESM states.
4. Keep rolling statistics rather than all per-step dictionaries in memory.
5. Move per-transition logs to DEBUG.
6. Set `episodes_per_epoch` explicitly instead of implicitly using all 7,701
   structures for every one of 32 epochs.

### Recommended next experiment

1. Do not resume from the corrupt replay snapshot. Keep `agent.pt` only for
   diagnostic inference.
2. First fix no-revisit masking, remaining-step state, and cycle-safe shaping.
3. Rebalance step and terminal scales and correct predicted-structure identity.
4. Fix replay/log/checkpoint memory behavior and use a new output directory.
5. Run a controlled 100k-300k-step experiment with fixed validation proteins,
   random baseline, and wild-type baseline.
6. Only then ablate `gamma=0.95-0.98`, a lower learning rate such as `3e-5`, and
   soft target-network updates. Lowering only `max_grad_norm` would hide the
   symptom rather than correct the target scale.

Acceptance criteria for the next run:

- Q stays near the empirical return scale and Q-target bias is near zero.
- Raw-gradient clipping is occasional rather than nearly universal.
- Two-step sequence reversals approach zero in the no-revisit base version.
- Terminal strength/toughness outperform random and wild-type baselines on fixed
  validation proteins.
- Predicted-structure reward is used only for exact terminal-sequence matches.

### Analysis artifacts

Detailed outputs are in:

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_fully-change-max-steps/analysis
```

Files:

- `analysis_summary.json`
- `optimization_bins.csv`
- `episode_bins.csv`
- `episode_action_cycles.csv`
- `action_cycle_phases.csv`
- `top_cycle_episodes.csv`
- `validation_checkpoints.csv`
- `optimization_diagnostics.png`
- `reward_policy_diagnostics.png`
- `analysis_report.md`

# 2026-07-28: four-GPU batch-128 training without gradient accumulation

### Existing capability check

The project already had a single-machine multi-GPU path:

```text
--mode multi --gpu-ids 0,1,2,3
```

It wraps the online and target Q networks with `torch.nn.DataParallel`.
However, the previous training script still used:

```text
micro_batch_size=4
gradient_accumulation_steps=4
effective_batch_size=16
```

Therefore the actual full-batch four-GPU path was not enabled by that script.

The DDQN optimization loop already supports a no-accumulation batch. When:

```text
micro_batch_size=128
gradient_accumulation_steps=1
```

the loop has one micro-batch, performs one backward call, and DataParallel
scatters the 128 samples across four GPUs as 32 samples per GPU.

### `training.py` changes

A standard full optimizer-batch argument was added:

```bash
--batch-size 128
```

This argument resolves internally to:

```text
micro_batch_size=128
gradient_accumulation_steps=1
effective_batch_size=128
```

It is intentionally different from `--train-batch-size`:

- `--train-batch-size` groups PDB episode paths in the dataset iterator;
- `--batch-size` controls the replay sample and DDQN optimizer batch.

The previous low-level `--micro-batch-size` and
`--gradient-accumulation-steps` arguments remain compatible.

A DataParallel batch-plan validator and logger were added. For the requested
configuration, `run_config.json` now records:

```json
{
  "parallel_batch_plan": {
    "gpu_count": 4,
    "effective_batch_size": 128,
    "batch_per_backward": 128,
    "gradient_accumulation_steps": 1,
    "per_gpu_batch_size": 32,
    "uneven_batch_remainder": 0
  }
}
```

Multi-GPU mode now rejects a batch-per-backward smaller than the GPU count and
warns when the batch cannot be divided evenly across devices.

### Four-GPU integration test

Added:

```text
tests/test_training_multi_gpu.py
```

The tests cover:

1. `--batch-size 128` disables gradient accumulation.
2. Four GPUs receive 32 samples each.
3. Invalid batches smaller than the GPU count are rejected.
4. A CUDA integration test creates a `[128, 16, 1280]` per-residue batch.
5. Forward hooks verify that `cuda:0`, `cuda:1`, `cuda:2`, and `cuda:3` all
   execute the Q-network forward pass.
6. The optimizer result reports `effective_batch_size=128` and
   `micro_batches=1`.
7. Hard target-network synchronization works after DataParallel wrapping.
8. Saved checkpoints remove DataParallel's `module.` prefix and remain
   compatible with ordinary agent loading.

Direct four-GPU result:

```text
1 passed in 6.24s
```

CPU/full test result:

```text
99 passed, 1 skipped in 9.91s
```

The skipped test is the four-GPU CUDA test when CUDA devices are not exposed to
the sandboxed CPU test process.

### Real CLI smoke test

`training.py` was also run through its actual multi-GPU CLI with:

```text
mode=multi
gpu_ids=[0, 1, 2, 3]
batch_size=128
micro_batch_size=128
gradient_accumulation_steps=1
per_gpu_batch_size=32
```

The smoke run completed one PyRosetta environment episode and successfully
wrote:

- `run_config.json`
- `agent_final.pt`
- `replay_buffer_final.npz`
- episode and step logs
- final diagnostic plots

Smoke output:

```text
/tmp/mprl-4gpu-cli-smoke-2
```

### Performance benchmark

A short synthetic benchmark compared equal optimizer batch size 128 under
three modes. AMP was enabled and the Q head used hidden dimensions `(256,256)`.

For residue length 128:

| Mode | Seconds/update | Samples/second | Relative to accumulation |
|---|---:|---:|---:|
| Single GPU, `32 x 4` accumulation | 0.0762 | 1678.7 | 1.00x |
| Single GPU, `128 x 1` full batch | 0.0621 | 2060.8 | 1.23x |
| Four GPU DataParallel, `128 x 1` | 0.0980 | 1305.7 | 0.78x |

For residue length 512:

| Mode | Seconds/update | Samples/second | Relative to accumulation |
|---|---:|---:|---:|
| Single GPU, `32 x 4` accumulation | 0.2734 | 468.3 | 1.00x |
| Single GPU, `128 x 1` full batch | 0.2474 | 517.4 | 1.11x |
| Four GPU DataParallel, `128 x 1` | 0.3658 | 349.9 | 0.75x |

The requested four-GPU mode is functionally correct, but current DataParallel
does not accelerate this particular Q network. The residue-wise Q head is
relatively small, so model replication, input scatter, output gather, and
cross-GPU gradient reduction cost more than the parallel computation saves.

There are two additional system-level limits:

1. PyRosetta mutation/repack and environment stepping are still serial.
2. ESM2 observation generation runs on the primary GPU rather than across four
   environment workers.

Also, batch 128 processes eight times as many replay samples per environment
step as the previous effective batch 16. Even with perfect scaling, an update is
not expected to take the same wall time as the old update.

Therefore:

- use the provided four-GPU script when batch-128 capacity or testing the
  multi-GPU path is the goal;
- if batch 128 fits on one RTX 4090, single-GPU `--batch-size 128` is faster in
  the current architecture;
- real four-GPU throughput scaling requires a later migration to one process
  per GPU with `DistributedDataParallel`, plus four parallel environment/replay
  data streams.

### Runnable script

Added:

```text
train_version_terminal_4gpu.sh
```

Run:

```bash
bash train_version_terminal_4gpu.sh
```

The script:

- exposes GPUs `0,1,2,3`;
- uses `--mode multi --gpu-ids 0,1,2,3`;
- uses `--batch-size 128`, so there is no gradient accumulation;
- enables AMP;
- writes the background process PID to `train.pid`;
- writes stdout/stderr to `train_stdout.log`;
- creates a timestamped output directory by default.

Optional overrides:

```bash
MPRL_VISIBLE_GPUS=0,1,2,3 \
MPRL_RUN_DIR=/path/to/output \
MPRL_PYTHON=/path/to/python \
bash train_version_terminal_4gpu.sh
```

# 2026-08-03: fix non-finite gradients in four-GPU AMP training

## Failure evidence

Analyzed run:

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
full_esm_terminal_4gpu_bs128_20260729_180841/train_stdout.log
```

The run completed 4,000 optimizer updates with finite metrics. The final
successful update reported:

```text
optimization_step=4000
loss=0.40741476
mean_q=11.07812500
mean_target=10.82310104
mean_abs_td=0.62820077
grad_norm=1.90614128
target_synced=True
```

The next replay batch failed before optimizer update 4,001 with:

```text
FloatingPointError: Gradient norm is NaN or infinity
```

No non-finite loss, Q value, target Q value, TD error, observation, or reward
was recorded in the preceding 4,000 updates. Target synchronization at update
4,000 is therefore temporal coincidence rather than the primary cause.

## Root cause

The old AMP implementation always used FP16 and PyTorch's default
`GradScaler`. In the installed PyTorch 2.12 environment, its defaults are:

```text
init_scale=65536
growth_factor=2
growth_interval=2000
backoff_factor=0.5
```

Consequently, the scale grows to 131,072 after update 2,000 and 262,144 after
update 4,000. The next backward pass can overflow FP16 even while the unscaled
loss and Q values remain finite. This exactly matches the failure boundary.

The original code detected the infinite norm after `unscale_()`, cleared the
gradients, and raised immediately. That prevented `GradScaler.update()` from
observing the overflow and lowering its scale, so PyTorch's intended AMP
recovery path could not run.

## Code changes

### `model/agent_module/ddqn_agent.py`

- Added `DDQNConfig.amp_dtype`, supporting `bfloat16` and `float16`; default is
  `bfloat16`.
- Added `DDQNConfig.amp_max_retries`, default 4.
- BF16 autocast does not use loss scaling because BF16 has the same exponent
  range as FP32 and avoids this FP16 scale overflow on supported GPUs.
- FP16 continues to use `GradScaler`. If the unscaled gradient norm is
  non-finite, the agent now updates the scaler, verifies that the scale was
  reduced, clears the invalid gradients, and recomputes the same replay batch.
- A recovered batch increments `optimization_steps` exactly once; failed
  attempts neither update parameters nor synchronize the target network.
- Added a separate pre-backward/optimizer guard for non-finite loss, Q, target,
  and TD metrics. These errors are not incorrectly classified as AMP overflow.
- `OptimizationResult` now records `amp_retries` and `amp_scale`.
- FP16 scaler state is saved and restored with checkpoints. Older checkpoints
  without this optional field remain loadable.
- BF16 requests fail early with a clear message on unsupported CUDA hardware.

### `training.py`

Added CLI options:

```text
--amp-dtype {bfloat16,float16}
--amp-max-retries N
```

The main optimization log now includes `amp_retries` and `amp_scale`. These
fields are also persisted by the existing optimization JSONL/TensorBoard path.

### `train_version_terminal_4gpu.sh`

The four-GPU batch-128 script now explicitly uses:

```bash
--use-amp --amp-dtype bfloat16
```

This is the recommended stable configuration for the current four RTX 4090
training host. FP16 remains available for comparison or older hardware.

## Tests

CPU and configuration regression tests:

```text
50 passed, 2 skipped in 14.33s
```

The skipped tests require CUDA. They were then executed directly with GPUs
0, 1, 2, and 3:

```text
2 passed in 4.46s
```

Coverage includes:

1. One four-GPU BF16 optimization with full batch 128 and no gradient
   accumulation.
2. Forward-hook confirmation that all four GPUs execute the Q head.
3. Finite loss and gradient, target synchronization, and checkpoint writing.
4. A forced FP16 overflow at the failed run's scale of 262,144.
5. Automatic scale recovery observed as:

```text
262144 -> 131072 -> 65536 -> 32768
```

The third retry completed successfully and produced one valid optimizer update.

## Conclusion and run method

The immediate failure was caused by FP16 dynamic-loss-scale growth, not by a
sudden invalid reward or Q-value divergence. The default four-GPU script now
avoids that failure mode with BF16, while the FP16 path has bounded automatic
recovery instead of terminating the run.

Start the corrected full run with:

```bash
bash train_version_terminal_4gpu.sh
```

During training, monitor `amp_retries`, `amp_scale`, `grad_norm`, `loss`, and
`mean_q_value`. For BF16, `amp_scale` is intentionally null and
`amp_retries` should stay at zero. A non-finite gradient under BF16 is treated
as a genuine numerical/model-data error rather than hidden by loss-scale
backoff.

# 2026-08-03: update frequency, episode horizon, and DDQN convergence analysis

## Main conclusion

The observation that reducing `max_steps` from 1024 to 50 and then to 24 lowers
the loss/gradient peak is reasonable, but it does **not** currently demonstrate
that a higher parameter-update frequency improves convergence.

The current training loop performs one optimizer update after every environment
transition once replay warmup is complete. Therefore, measured in optimizer
updates per environment transition, the update-to-data ratio is approximately
1 for all three `max_steps` settings.

Changing `max_steps` instead changes several other parts of the learning
problem simultaneously:

1. terminal-reward density in replay;
2. Bellman backup horizon and discounted return scale;
3. the number of different proteins represented in a fixed transition budget;
4. the number of completed episodes seen before replay warmup and epsilon decay;
5. the maximum length of mutation/reversal cycles;
6. how often validation, plots, and checkpoints run when they are scheduled by
   episode count.

The improvement is therefore real, but its current interpretation should be:
**a shorter and better-conditioned finite-horizon task is easier for the current
one-step DDQN to fit**. It is not yet an isolated update-frequency result.

## What the current code actually does

In `training.py`, every accepted or rejected environment action is immediately:

1. added to replay;
2. followed by `agent.optimize_from_replay_buffer(...)`;
3. sampled as part of one replay batch when the buffer has at least 512 rows.

Thus, after warmup:

```text
optimizer updates / environment transitions ~= 1
```

In addition, `ReplayBuffer` stores `done = terminated OR truncated`, and DDQN
does not bootstrap when `done=True`. Because `MechanicalProteinEnv` truncates at
`max_steps` and adds terminal mechanical reward on that transition, changing
`max_steps` changes the Bellman objective itself. It is not merely a rollout
boundary or dataloader parameter.

This finite-horizon formulation is valid, but the remaining step budget must
eventually be included in the observation; otherwise identical sequence
embeddings at different remaining horizons have different correct Q values.

## Quantitative effect of `max_steps`

The following estimates use the current `gamma=0.99`, replay batch size 128,
and assume episodes usually reach the configured horizon.

| `max_steps` | Discounted step-weight sum | Terminal weight at episode start | Expected terminal rows per batch 128 | Probability batch contains terminal reward | Episodes before epsilon reaches minimum |
|---:|---:|---:|---:|---:|---:|
| 1024 | 99.997 | 0.000034 | 0.125 | 11.8% | 48.8 |
| 50 | 39.499 | 0.611 | 2.56 | 92.5% | 1,000 |
| 24 | 21.432 | 0.794 | 5.33 | 99.6% | 2,083 |

Calculations:

```text
discounted step-weight sum = (1 - gamma^H) / (1 - gamma)
terminal weight at start  = gamma^(H - 1)
expected terminal rows    = batch_size / H
P(any terminal row)       = 1 - (1 - 1/H)^batch_size
episodes before eps_min   = epsilon_decay_steps / H
```

This explains much of the observed behavior:

- At `H=1024`, terminal mechanical reward is almost invisible from the start of
  an episode and most replay batches contain no terminal transition.
- At warmup step 512, `H=1024` has not completed even one episode, while `H=50`
  and `H=24` have produced about 10 and 21 terminal transitions respectively.
- With the previously observed mean positive step reward near 0.67, the
  approximate discounted step-return baselines are 67.0, 26.5, and 14.4 for
  horizons 1024, 50, and 24. Shortening the horizon directly reduces the scale
  that Q and TD targets must learn.
- A 50,000-transition replay window covers only about 49 full `H=1024`
  trajectories, versus about 1,000 at `H=50` and 2,083 at `H=24`. The shorter
  run mixes many more proteins and fewer long correlated trajectories.
- One-step TD has to propagate delayed terminal reward through at most 24
  transitions instead of 1024. This is a major reduction in credit-assignment
  difficulty.

Therefore, the lower loss peak at `H=24` may partly be a change in target scale,
not necessarily a better mechanical-protein policy. Policy quality still needs
to be judged by fixed validation proteins and terminal mechanical-property
improvement over wild type/random policy.

## Update-frequency improvement space

There is meaningful room to improve the update schedule. The most important
issue is that the four-GPU run changed replay batch size from effective batch 16
to batch 128 while still updating after every new transition.

Define:

```text
F = newly collected transitions between optimizer events
G = gradient steps at each optimizer event
B = replay batch size
gradient UTD               = G / F
sample-consumption ratio   = B * G / F
```

Current configurations are approximately:

| Configuration | B | F | G | Gradient UTD | Replay samples trained per new transition |
|---|---:|---:|---:|---:|---:|
| Single GPU, accumulated | 16 | 1 | 1 | 1.0 | 16 |
| Multi-GPU full batch | 128 | 1 | 1 | 1.0 | 128 |

The batch-128 run therefore consumes eight times as many replay examples for
each newly collected transition. Although its formal gradient UTD is still 1,
its replay reuse pressure is much higher. This can fit a small, stale, and
highly correlated replay distribution too aggressively, increase Q-target
feedback, and produce the observed early rise in TD loss and gradient norm.

### Recommended first update-frequency experiment

Keep `max_steps=24` and batch size 128 fixed, then compare:

| Experiment | Collect frequency F | Gradient steps G | Gradient UTD | Samples/new transition |
|---|---:|---:|---:|---:|
| A | 1 | 1 | 1.0 | 128 |
| B | 4 | 1 | 0.25 | 32 |
| C | 8 | 1 | 0.125 | 16 |

Experiment C is the cleanest first candidate because it restores the same
sample-consumption ratio as the old effective-batch-16 training while retaining
the batch-128 four-GPU update.

The future implementation can expose two independent options:

```text
--train-frequency F
--gradient-steps G
```

The training loop would collect `F` transitions, then perform `G` replay
updates. This is better than overloading `max_steps`, which controls the
biological optimization horizon and terminal-reward timing.

### Controls required for a valid comparison

1. Compare runs at equal environment-transition budgets, not equal epoch or
   episode counts.
2. Use at least three random seeds; one RL curve is not sufficient to rank
   schedules.
3. Increase replay warmup from 512 to at least 4,096 for batch 128, or define
   warmup using both transition count and completed-protein count.
4. Keep target-network refresh constant in **environment steps**. If update
   frequency changes but `target_sync_interval=500` remains measured in
   optimizer steps, target staleness changes and confounds the experiment.
5. Run epsilon decay, validation, checkpointing, and plotting on explicit
   global environment-step schedules. Their current episode-based frequencies
   correspond to very different transition intervals at H=24, 50, and 1024.
6. Use identical train/validation indices, model initialization, reward scales,
   and replay capacity.

The current multi-GPU implementation parallelizes Q-network computation through
`DataParallel`; it does not collect experience from multiple parallel
environments. Consequently, more GPUs currently increase optimization capacity
but do not increase the rate or diversity of new transitions. Parallel
environment workers are a later way to improve this balance.

## Why loss and gradient can rise before falling

This shape is not automatically a training failure. In DDQN, the regression
target is itself moving:

1. The network starts with Q values near zero while rewards are mostly positive.
2. As positive rewards bootstrap through the target network, both Q and target
   scale rise.
3. Epsilon decay changes the replay distribution from broad exploration toward
   the current greedy policy.
4. Hard target synchronization every 500 optimizer updates introduces discrete
   target changes.
5. Once replay composition, epsilon, and Q scale change more slowly, TD loss can
   decline because the network catches up with its targets.

This means falling TD loss only proves improved consistency with the current
bootstrapped targets. It does not prove that predicted structures have better
mechanical properties. A degenerate cyclic policy can also have low TD loss.

## Other important influences

### 1. Positive step-reward baseline and horizon-dependent objective

The current step reward is a mean of four 0-1 scores and is usually positive.
Longer episodes therefore accumulate a large survival-like reward even without
terminal mechanical improvement. Recommended directions are:

- potential-based shaping: `Phi(next) - Phi(current)`;
- center neutral step components around zero;
- or scale each step contribution by `1 / max_steps` so its total episode mass
  is comparable across horizons.

Potential-based shaping is preferred because immediate reversals then have
approximately zero net shaping reward.

### 2. Partial observability

The Q network sees ESM2 sequence embeddings, but transition and reward also
depend on the current packed 3D structure, mutation history, visited positions,
and remaining steps. The same sequence can therefore receive contradictory TD
targets. Longer episodes amplify this state aliasing.

Add at least normalized remaining budget and mutation-history/revisit features.
Structural summary features or a structure encoder should later represent the
PyRosetta state used by the reward function.

### 3. Repeated mutations and two-step reversals

Previous analysis found severe revisits and immediate reversals with
`prevent_revisit_positions=false`. Reducing `max_steps` limits how long the
cycle can continue but does not remove the policy defect. The next base
experiment should enable no-revisit masking, then separately study whether
controlled revisits are useful.

### 4. One-step credit assignment

The mechanical predictor is primarily terminal reward. One-step DDQN propagates
that signal backward slowly. Reasonable follow-up methods are 3-10 step returns,
TD(lambda), or a replay stratum that guarantees a small terminal-transition
fraction. Prioritized replay is possible, but requires importance weights and a
cap because large noisy TD errors can otherwise worsen overestimation.

### 5. Target network and Q overestimation

Double DQN reduces but does not eliminate overestimation, especially with a
large variable action space, max selection, positive rewards, and partial
observability. Compare hard sync against Polyak/soft target updates, and log
`mean_q - mean_target` plus Q versus empirical episode return. Any target-update
ablation must be normalized to environment steps.

### 6. Replay composition

Uniform transition replay overrepresents whatever trajectories the current
policy recently generates. It does not ensure balanced protein families,
episode stages, sequence lengths, or terminal transitions. Consider stratified
sampling by protein/episode stage and report the replay terminal fraction,
unique-protein count, and sample age.

### 7. Exploration schedule and dataset coverage

Epsilon currently decays over 50,000 environment steps. That corresponds to
only about 49 proteins at H=1024 but about 2,083 at H=24. Exploration should be
defined relative to dataset coverage or number of completed protein episodes,
or this difference must be acknowledged in comparisons.

### 8. Batch size, learning rate, and gradient clipping

Batch 16 and batch 128 are different optimization regimes. The same learning
rate may work, but must be ablated rather than assumed. Track the fraction of
updates whose **pre-clipping** gradient norm exceeds 10; a declining loss while
almost every update is clipped is not a healthy convergence signal.

### 9. Terminal predictor validity

The random-forest mechanical predictor can be OOD on heavily mutated or poorly
packed structures. The predicted-structure branch must match the exact terminal
sequence; a source-sequence PDB should not be averaged as if it were the mutated
terminal structure. Log predictor uncertainty/proximity to training support,
exact-sequence match coverage, and the two structure-specific rewards.

### 10. Comparability and logging

Curves should share three x-axes: environment steps, optimizer updates, and
completed proteins. Report rolling means with the same window in environment
steps. The current `validate_every=240` episodes evaluates every 245,760 steps
at H=1024, 12,000 at H=50, and 5,760 at H=24, so validation curves indexed only
by episode are not directly comparable.

## Proposed experiment order

1. Keep `max_steps=24`; compare batch-128 train frequencies 1, 4, and 8 over the
   same 100k-300k environment-step budget.
2. Select the schedule with finite Q/gradients, low clipping fraction, and the
   best fixed-validation terminal improvement, not merely the lowest TD loss.
3. Enable no-revisit masking and add remaining-step budget to the observation.
4. Center or potential-shape step reward, then compare H=24 and H=50 at equal
   environment steps and equal reward scale.
5. Only after those controls, ablate `gamma` (for example 0.95, 0.97, 0.99),
   soft target updates, and n-step returns.

Primary metrics should be:

- fixed-validation terminal strength/toughness delta versus wild type;
- advantage over random-policy and greedy-step-reward baselines;
- Q versus empirical return calibration and `mean_q - mean_target`;
- TD loss, raw gradient norm, and clipping fraction;
- terminal-transition fraction and unique proteins in replay;
- revisit, immediate-repeat, and two-step-reversal rates;
- performance across at least three seeds.

## Configuration note

At the time of this analysis, `train_version_terminal_4gpu.sh` exposes four
physical GPUs through `CUDA_VISIBLE_DEVICES=0,1,2,3` but passes
`--gpu-ids 1,2`, so the current script uses two visible devices, not four. This
does not explain the common rise-then-fall loss pattern, but the run should be
labelled as two-GPU unless the IDs are changed back to `0,1,2,3`.

# 2026-08-03: independent collection and gradient-update schedules

## Implemented CLI options

`training.py` now supports two independent parameters:

```text
--train-frequency F
--gradient-steps G
```

Their behavior is:

1. collect `F` new environment transitions;
2. trigger one replay-training event;
3. execute `G` optimizer updates during that event;
4. independently resample replay for every optimizer update.

The defaults are:

```text
--train-frequency 1
--gradient-steps 1
```

These defaults preserve the previous update-after-every-transition behavior.
Both arguments must be positive integers.

The effective schedule is reported as:

```text
gradient updates per transition = G / F
replay samples per transition   = batch_size * G / F
```

`run_config.json` now includes an `update_schedule` object containing:

- `train_frequency`;
- `gradient_steps`;
- `gradient_updates_per_transition`;
- `replay_samples_per_transition`.

Each optimization JSONL/TensorBoard record also includes:

- `optimizer_event`;
- `completed_environment_steps`;
- `gradient_step_in_event`;
- `gradient_steps_requested`;
- `train_frequency`.

This makes `G > 1` runs unambiguous: several optimization rows may share the
same environment step but have different optimizer-step and event-step IDs.

Training events that occur before replay warmup are skipped. The code does not
perform catch-up updates for skipped events, because the corresponding replay
distribution was not yet valid.

## Code organization

The scheduling implementation in `training.py` is separated into testable
helpers:

```text
update_schedule_summary(...)
should_run_optimizer_event(...)
run_replay_optimization_event(...)
```

The environment remains responsible only for transition generation, while
DDQN remains responsible only for one replay optimizer update. Collection and
update cadence are controlled in the root training loop.

## Multi-GPU launcher controls

`train_version_terminal_4gpu.sh` now accepts these environment overrides:

```text
MPRL_GPU_IDS
MPRL_TRAIN_FREQUENCY
MPRL_GRADIENT_STEPS
MPRL_REPLAY_WARMUP_SIZE
MPRL_TARGET_SYNC_INTERVAL
MPRL_RUN_LABEL
MPRL_RUN_DIR
```

Example custom schedule:

```bash
MPRL_TRAIN_FREQUENCY=4 \
MPRL_GRADIENT_STEPS=2 \
MPRL_TARGET_SYNC_INTERVAL=250 \
bash train_version_terminal_4gpu.sh
```

The launcher prints the resolved GPU IDs, collection/update schedule, replay
warmup, output directory, and target-sync interval before returning.

## One-command comparison scripts

Added three launchers:

```text
train_update_frequency_f1.sh
train_update_frequency_f4.sh
train_update_frequency_f8.sh
```

All three keep these controls fixed:

- batch size: 128;
- gradient steps per event: 1;
- `max_steps`: 24;
- replay warmup: 4,096 transitions;
- epochs, dataset indices, seed, reward, ESM2 model, and BF16 AMP settings;
- target synchronization at approximately every 500 environment transitions.

The comparison matrix is:

| Script | F | G | Gradient UTD | Replay samples/new transition | Target sync interval in optimizer steps | Approximate target sync interval in environment steps |
|---|---:|---:|---:|---:|---:|---:|
| `train_update_frequency_f1.sh` | 1 | 1 | 1.0 | 128 | 500 | 500 |
| `train_update_frequency_f4.sh` | 4 | 1 | 0.25 | 32 | 125 | 500 |
| `train_update_frequency_f8.sh` | 8 | 1 | 0.125 | 16 | 63 | 504 |

Run the experiments individually:

```bash
bash train_update_frequency_f1.sh
bash train_update_frequency_f4.sh
bash train_update_frequency_f8.sh
```

Each command starts a timestamped background run and writes its PID and stdout
log inside its own output directory under:

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
```

Do not launch all three simultaneously on the same GPU IDs; doing so would make
throughput, memory pressure, and training stability incomparable. Run them
sequentially, or assign non-overlapping GPUs explicitly.

The current base launcher defaults to visible GPU IDs `1,2`. To use all four
visible devices for any comparison run:

```bash
MPRL_GPU_IDS=0,1,2,3 bash train_update_frequency_f8.sh
```

Use the same GPU selection for every compared run.

## Verification

Python scheduling, DDQN, and logging tests:

```text
60 passed, 2 CUDA-only tests skipped in 5.68s
```

The tests cover:

1. optimizer-event boundaries for F=1, 4, and 8;
2. independent UTD/sample-ratio calculation for arbitrary F and G;
3. exactly G calls per replay-training event;
4. stopping an event cleanly while replay is warming up;
5. a real CPU DDQN/replay event with `G=3`, producing optimizer steps 1, 2,
   and 3;
6. the existing DDQN, logger, AMP, and multi-GPU regression paths.

All four shell launchers pass `bash -n`, and `training.py --help` exposes both
new options.

A complete CLI smoke test used:

```text
max_steps=4
train_frequency=2
gradient_steps=2
replay_warmup_size=2
target_sync_interval=2
```

Observed result:

```text
environment transitions: 4
optimizer events: 2
gradient steps/event: 2
total optimizer updates: 4
target sync updates: 2 and 4
exit status: 0
```

Smoke artifacts are stored at:

```text
/tmp/mprl-train-frequency-smoke-dHzp2L
```

# 2026-08-04: non-finite forward metrics in the F/G comparison runs

## Runs analyzed

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
update_frequency_f1_g1_bs128_h24_20260803_175427
update_frequency_f4_g1_bs128_h24_20260803_175439
update_frequency_f8_g1_bs128_h24_20260803_175444
```

All three runs ended with:

```text
FloatingPointError: DDQN forward or TD metrics contain NaN or infinity
before the optimizer step.
```

The guard behaved correctly: it stopped before applying the failing optimizer
update, and the final checkpoints contain finite online-network parameters,
target-network parameters, and Adam states.

## Failure boundary

| Run | Last successful environment step | Last successful optimizer step | Epsilon | Last loss | Last absolute TD error | Last raw grad norm | Last mean Q | Last mean target Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F=1, G=1 | 24,605 | 20,510 | 0.533 | 54.06 | 54.55 | 111.66 | 472.5 | 460.6 |
| F=4, G=1 | 39,148 | 8,764 | 0.256 | 44.78 | 45.28 | 50.43 | 491.0 | 480.9 |
| F=8, G=1 | 33,488 | 3,675 | 0.364 | 57.60 | 58.10 | 64.80 | 511.5 | 488.8 |

The failing events were the immediately following scheduled updates:

```text
F1: environment step 24,606, optimizer step 20,511
F4: environment step 39,152, optimizer step 8,765
F8: environment step 33,496, optimizer step 3,676
```

All three runs therefore fail near the same Q scale, about 470-510, despite
reaching that scale after very different numbers of optimizer/environment
steps. This common boundary is more informative than the raw failure time.

## What is and is not non-finite

Checkpoint audit after the exception:

| Run | Non-finite online tensors | Non-finite target tensors | Non-finite Adam tensors | Maximum absolute online parameter |
|---|---:|---:|---:|---:|
| F1 | 0 | 0 | 0 | 0.895 |
| F4 | 0 | 0 | 0 | 0.430 |
| F8 | 0 | 0 | 0 | 0.291 |

Logged transition data are also finite:

| Run | Step/total reward range | Terminal reward range | Sequence-length range |
|---|---:|---:|---:|
| F1 | -0.584 to 2.751 | -1.088 to 2.107 | 28-584 |
| F4 | -0.564 to 2.911 | -1.075 to 2.412 | 26-584 |
| F8 | -0.535 to 2.736 | -1.065 to 2.040 | 26-584 |

`DDQNAgent._coerce_replay_batch()` rejects NaN/Inf observations, next
observations, and rewards before network execution. The immediate error is
therefore not evidence of a corrupt reward row or a permanently NaN model.

The likely immediate mechanism is a batch-specific BF16 forward/target
overflow or invalid activation after Q values have already diverged. The mean
metrics hide per-action/per-sample extremes, and the mixed replay contains
proteins with approximately 520 to more than 11,000 valid mutation actions.
A particular long-protein batch can expose a much larger max-Q or hidden
activation than the preceding batch mean.

This is different from the earlier FP16 GradScaler failure:

- these runs use BF16;
- BF16 has no GradScaler or scale-backoff path;
- the exception is raised before gradient-norm calculation;
- retrying a loss scale cannot solve it.

## Primary root cause: Q-value divergence

With `max_steps=24`, normalized local rewards, and observed terminal rewards,
the finite-horizon return should be on the order of tens, not approximately
500. The three networks are already severely overestimating Q before the first
non-finite batch.

Evidence across all successful updates:

| Run | Maximum loss | Maximum raw grad norm | Fraction grad norm > 10 | Last-500 mean Q-target bias |
|---|---:|---:|---:|---:|
| F1 | 69.82 | 230.29 | 55.6% | +16.23 |
| F4 | 66.38 | 151.37 | 62.7% | +12.63 |
| F8 | 83.00 | 133.63 | 63.5% | +7.16 |

Gradient clipping prevents one very large update, but more than half of all
updates are already in the clipped regime. It does not correct an inflated
Bellman target.

The main reinforcing factors are:

1. mostly positive step rewards with `gamma=0.99`;
2. bootstrapping through a moving target network;
3. max selection across a length-dependent action space of up to 11k+ actions;
4. sequence-only ESM observations while reward/transition also depend on packed
   structure, mutation history, action mask, and remaining horizon;
5. repeated mutation/reversal cycles because no-revisit masking is disabled;
6. raw ESM embeddings entering an MLP without explicit input/hidden
   normalization;
7. hard target copies that can rapidly feed overestimated online Q values back
   into TD targets.

The failure guard is therefore the last symptom detector. Simply removing the
guard, skipping the batch, or increasing the gradient clip threshold would let
the divergence continue.

## Correct interpretation of the F/G result

The statement "larger F/G grows faster" is true only when curves are indexed by
optimizer step. It is not true when indexed by collected environment data.

Recent 200-update mean Q at equal environment steps:

| Environment step | F1 | F4 | F8 |
|---:|---:|---:|---:|
| 8,000 | 9.48 | 6.55 | 5.52 |
| 16,000 | 105.68 | 35.62 | 39.58 |
| 24,000 | 440.75 | 109.03 | 147.69 |

At equal environment experience, F1 generally grows fastest because it applies
the most optimizer updates.

At optimizer step 1,000:

| Run | Environment steps already collected | Mean Q near optimizer step 1,000 |
|---|---:|---:|
| F1 | 5,095 | 1.38 |
| F4 | 8,092 | 6.69 |
| F8 | 12,088 | 18.00 |

Thus F8's 1,000th optimizer update is not comparable with F1's 1,000th update:
it has seen substantially more transitions, more proteins, lower epsilon, and
more terminal rewards.

There is an additional target-network confound. The comparison scripts used:

```text
F1 target_sync_interval=500 optimizer steps
F4 target_sync_interval=125 optimizer steps
F8 target_sync_interval=63 optimizer steps
```

This kept target sync near every 500 environment transitions, but it means that
by optimizer step 1,000 the target network was copied about 2, 8, and 15 times
respectively. F8 therefore has much less target lag per optimizer update, which
accelerates the feedback of inflated Q estimates. The current result combines
update frequency and target-network cadence; it is not a clean one-variable
ablation.

## Simultaneous-run confound

The three jobs started within 17 seconds of each other and all configured GPU
IDs `0,1,2,3`. They therefore competed for the same GPU memory, compute, ESM2
encoding, and DataParallel communication.

This does not explain why all three reach Q approximately 500, but it makes
wall-clock speed, throughput, kernel scheduling, and numerical reproducibility
invalid for comparison. Future runs must be sequential or use non-overlapping
GPU groups.

## Policy/mechanical-reward result

The local step/episode return shows a small upward tendency in F4/F8, but the
terminal mechanical reward does not show meaningful learning:

| Run | Terminal reward vs global-step correlation | First-200 mean terminal reward | Last-200 mean terminal reward | Validation reward trend correlation |
|---|---:|---:|---:|---:|
| F1 | -0.033 | -0.101 | -0.158 | -0.088 |
| F4 | +0.003 | -0.101 | -0.084 | -0.204 |
| F8 | -0.033 | -0.104 | -0.194 | +0.035 |

None of the three runs provides evidence that the policy improved the intended
terminal mechanical property before Q divergence. Lower TD loss at an early
stage should not be treated as model selection evidence.

F4 survived the largest number of environment transitions, but this is only a
numerical observation. It is not yet the best biological policy.

## Proposed solution

### Phase 1: identify the immediate numerical boundary

Run one job only, starting from a new model and replay buffer:

1. choose F=4, G=1 as the diagnostic schedule;
2. keep all current settings initially, but disable AMP and run the Q head in
   FP32;
3. log per-batch and per-sample `max_abs_input`, hidden activation maxima,
   `max_abs_online_q`, `max_abs_target_q`, selected Q/target ranges, sequence
   lengths, valid-action counts, and replay indices;
4. keep the exact failed-batch diagnostics before raising;
5. run beyond 40k environment steps or until the same Q scale is reached.

Interpretation:

- if FP32 remains finite beyond the BF16 failure point while Q continues to
  grow, BF16 is the immediate overflow mechanism but Q divergence remains the
  algorithmic problem;
- if FP32 also becomes non-finite, the failure is fully algorithmic;
- if only particular sequence lengths/input norms fail, add input normalization
  and length-stratified diagnostics before changing the RL schedule.

An optional recovery mechanism can recompute a failing BF16 batch once in FP32.
It should save diagnostics and continue only when the FP32 result is finite.
Blindly skipping non-finite batches would bias replay and conceal the cause.

### Phase 2: constrain Bellman/Q scale

Recommended changes, in priority order:

1. Center step shaping around zero or replace it with potential differences,
   rather than giving a positive baseline every step.
2. Normalize the complete reward, including terminal reward, to a documented
   bounded range such as [-1, 1].
3. Add normalized remaining-step budget and mutation-history/no-revisit state to
   the observation; enable `prevent_revisit_positions` for the base experiment.
4. Add LayerNorm to the ESM per-residue input/head and record embedding norms by
   protein.
5. Lower learning rate from `1e-4` to `3e-5` as a controlled ablation.
6. Compare `gamma=0.95`, `0.97`, and `0.99` after reward centering.
7. Use a slower target update: initially keep hard sync at 500 **optimizer
   steps** for every F, then evaluate a Polyak target update separately.
8. Consider 3-10 step returns after the one-step baseline is numerically stable,
   so terminal reward propagates without requiring large bootstrapped Q chains.

Because H=24 and reward bounds are known, a theoretical finite-horizon Q range
can be calculated and monitored. Target/Q clipping to that range can be used as
an emergency safeguard, but only after reward semantics are finalized; otherwise
it silently changes the objective.

### Phase 3: rerun the F/G comparison cleanly

After Phase 1/2 produces a stable baseline:

1. run F=1, 4, and 8 sequentially;
2. use the same target interval of 500 optimizer steps in the first clean
   comparison;
3. stop each run at the same environment-step budget, for example 50k or 100k;
4. report both environment-step and optimizer-step axes;
5. use at least three seeds;
6. select by fixed-validation terminal mechanical improvement over wild type
   and random policy, not by minimum training loss.

The clean comparison should report:

- mean/max Q and target Q, plus Q-target bias;
- Q versus empirical finite-horizon return calibration;
- loss, TD error, raw grad norm, and clipping fraction;
- terminal reward and strength/toughness deltas;
- sequence-length/action-count stratified results;
- replay sample age and terminal-transition fraction;
- revisit and reversal rates.

## Operational notes

- Do not resume these three final checkpoints for continued training. They are
  finite and useful for diagnosis, but their Q functions are already badly
  overestimated.
- Each numerical failure triggered final replay serialization of approximately
  31 GB, delaying the visible traceback by roughly 20-35 minutes. A future
  failure path should save a small diagnostic checkpoint first and make the
  full replay snapshot optional/atomic.
- The final log line `Training finished` is emitted from cleanup even when the
  original exception is re-raised afterward. Run status should distinguish
  successful completion from failed finalization to avoid misleading monitoring.

# 2026-08-04: independent F4 rerun confirms reproducible Q divergence

## Run analyzed

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
update_frequency_f4_g1_bs128_h24_20260804_094941
```

This job was run independently, so the previous overlap of three jobs on GPUs
0-3 is no longer a possible explanation for the failure.

## Configuration audit

The rerun used:

```text
train_frequency=4
gradient_steps=1
batch_size=128
max_steps=24
replay_warmup_size=4096
target_sync_interval=125 optimizer steps
gamma=0.99
learning_rate=1e-4
use_amp=true
amp_dtype=bfloat16
prevent_revisit_positions=false
```

This is an independent repeat of the original F4 configuration. It is **not**
the previously proposed FP32 diagnostic run: `train_version_terminal_4gpu.sh`
still unconditionally passes `--use-amp --amp-dtype bfloat16`, and the F4
wrapper still sets target sync to 125 optimizer steps.

Therefore, this run successfully tests reproducibility, but it does not yet
separate BF16's immediate numerical boundary from the underlying Q divergence.

## Independent reproduction

| Metric | Previous F4 run | Independent F4 rerun | Difference |
|---|---:|---:|---:|
| Failing environment event | 39,152 | 38,680 | -1.2% |
| Next optimizer step that failed | 8,765 | 8,647 | -1.3% |
| Last mean Q | 491.0 | 486.0 | -1.0% |
| Last mean target Q | 480.9 | 474.0 | -1.4% |
| Maximum loss | 66.38 | 66.61 | +0.3% |
| Maximum raw grad norm | 151.37 | 151.54 | +0.1% |
| Fraction of updates with grad norm > 10 | 62.70% | 62.55% | -0.15 percentage points |

The full growth trajectory is also nearly identical:

| Environment step | Previous mean Q | Independent mean Q | Previous loss | Independent loss |
|---:|---:|---:|---:|---:|
| 8,000 | 6.55 | 6.45 | 0.279 | 0.274 |
| 16,000 | 35.62 | 36.64 | 2.444 | 2.499 |
| 24,000 | 109.03 | 112.08 | 9.017 | 9.485 |
| 32,000 | 255.50 | 262.88 | 22.068 | 22.859 |

This close reproduction rules out simultaneous GPU contention as the primary
cause. It also makes a random isolated bad reward or a one-off replay accident
unlikely as the root cause. A particular sampled batch is the final trigger,
but the network reaches that trigger only after a deterministic Q-growth phase.

## State after the exception

The failing update was scheduled at completed environment step 38,680. The
preceding successful update reported:

```text
optimization_step=8646
loss=36.7136
mean_absolute_td_error=37.2128
grad_norm=58.5433
mean_q=486.0
mean_target_q=473.98
```

Post-exception checkpoint audit:

```text
online non-finite tensors: 0
target non-finite tensors: 0
Adam non-finite tensors: 0
maximum absolute online parameter: 0.487
maximum absolute target parameter: 0.487
```

All recorded rewards were finite:

```text
total/step reward range: -0.563 to 3.062
terminal reward range: -1.068 to 2.562
sequence length range: 26 to 584
```

The failed batch preview contained ordinary sequence lengths between 48 and
151, not only the longest proteins. Only the first ten replay indices are
currently logged, so the exact offending row cannot be identified from this
run. The current environment sequence at the failure time is also not
necessarily the cause because the optimizer samples an unrelated replay batch.

The checkpoint remains finite because the forward-metric guard raises before
backward/Adam. This confirms the guard is preventing permanent model
contamination.

## Root-cause conclusion

The root cause is reproducible Bellman/Q-value divergence. BF16 or a particular
mixed replay batch is the immediate mechanism that converts an already inflated
Q function into a non-finite forward metric.

For the observed mean step reward of approximately 0.623 and H=24, an expected
discounted local-reward return is on the order of 10-20, with a terminal
adjustment on the order of one. A mean Q near 486 is not a plausible calibrated
finite-horizon value.

The independent run also shows:

```text
last-500 mean Q-target bias: +12.77
updates clipped at grad norm 10: 62.55%
terminal reward/global-step correlation: +0.0059
first-200 mean terminal reward: -0.101
last-200 mean terminal reward: -0.072
```

The terminal reward has no meaningful trend, while Q, TD error, and gradient
norm increase smoothly and reproducibly. The model is fitting an expanding
bootstrapped target rather than improving the terminal mechanical objective.

## Why the same failure repeats

1. Positive local reward provides a baseline return even for mechanically
   unproductive cycles.
2. `gamma=0.99` preserves almost all one-step target inflation.
3. Hard target sync every 125 optimizer steps repeatedly copies inflated online
   estimates into the bootstrap target. The run performs about 69 such copies
   before failure.
4. Greedy selection takes a maximum across a protein-length-dependent action
   space, increasing extreme-value overestimation.
5. ESM sequence embeddings omit remaining time, current packed structure, and
   mutation history, so the same encoded state can have inconsistent targets.
6. Revisits remain enabled; the final episode visibly mutates position 1 through
   several amino acids in consecutive steps, illustrating the cycle problem.
7. Once activations/Q values are large, one otherwise finite replay batch can
   overflow or produce an invalid BF16 forward statistic.

## Recommended solution sequence

### 1. Do not run the same F4 wrapper again

Another unchanged rerun is expected to fail near 38k-39k environment steps and
Q approximately 500. Do not resume either divergent F4 checkpoint for training.

### 2. Make the FP32 diagnostic executable

The launcher needs an AMP on/off control, for example:

```text
MPRL_USE_AMP=0
```

or a dedicated FP32 diagnostic script. The diagnostic must keep every other F4
setting unchanged initially, including target sync 125, so that precision is
the only changed variable.

Before this run, add diagnostics for:

- complete replay indices, sequence lengths, and valid-action counts;
- input max/mean/L2 norm per sample;
- max/min/finite counts for online Q, masked online Q, target Q, selected Q,
  TD target, TD error, and loss;
- hidden activation maxima for each linear/ReLU block;
- Q statistics split by protein-length/action-count bins.

Run until at least 40k environment steps or until mean/max Q crosses the BF16
failure range. FP32 may prevent the immediate NaN, but it is not considered a
solution if Q continues toward 500.

### 3. Add an early numerical stop condition

Do not wait for NaN. Once reward semantics are bounded, calculate an expected
finite-horizon Q range and stop/save diagnostics when mean or max Q exceeds a
conservative multiple of that range. For the current diagnostic, a warning near
Q=50 and hard diagnostic stop near Q=100 would expose divergence hours earlier.

### 4. Stabilize one component at a time

Recommended ablation order after the FP32 precision test:

1. Keep F4/G1 and change target sync from 125 to 500 optimizer steps.
2. If Q still grows, reduce learning rate from `1e-4` to `3e-5`.
3. Center the step reward or use potential-based reward differences; normalize
   total reward, including terminal reward, to a fixed documented range.
4. Enable no-revisit masking and include normalized remaining horizon in the
   observation.
5. Add LayerNorm/input normalization before the per-residue Q head.
6. After the one-step baseline is stable, test lower gamma and n-step returns.

For scientific attribution, run these as sequential one-change ablations. For a
fast engineering stabilization baseline, target sync 500, learning rate 3e-5,
centered reward, no revisits, remaining-horizon state, and input LayerNorm can be
combined, but the individual cause will then be less identifiable.

### 5. Only then return to F/G comparison

The F1/F4/F8 comparison should wait until one configuration remains calibrated
for at least 50k-100k environment steps. Then compare sequentially, with:

- identical optimizer-step target-sync interval;
- identical AMP/precision and learning rate;
- equal environment-step budgets;
- at least three seeds;
- model selection based on validation terminal mechanical reward rather than
  training TD loss.

## Operational issue confirmed again

The exception occurred at 14:47:33, but the traceback was not printed until
15:21:48 because finalization serialized a 36 GB replay buffer and generated
plots first. The run directory also contains a separate 32 GB periodic replay
snapshot. Numerical-failure handling should avoid writing another full replay
unless explicitly requested; save the small model plus failing-batch diagnostic
record first.

# 2026-08-06: FP32 F/G comparison, convergence judgment, and next steps

## Scope and configuration audit

This analysis used snapshots of the three logs at approximately 2026-08-06
10:43. All three jobs were still active and none contained a non-finite error at
the snapshot boundary.

The directory described in the task as F8/G2 is actually an **F8/G1** run. Its
directory name is `update_frequency_f8_g1_bs128_h24_20260805_150118`, and
`run_config.json` confirms `train_frequency=8` and `gradient_steps=1`. It must
not be interpreted as an F8/G2 result.

| Run | Precision | Devices/batch | F/G | Warmup | Target sync | Environment steps | Optimizer steps |
|---|---|---|---:|---:|---:|---:|---:|
| F4/G1 | FP32 | 4 GPU, batch 128 | 4/1 | 4,096 | 125 optimizer steps | 198,264 | 48,543 |
| F8/G1 | FP32 | 4 GPU, batch 128 | 8/1 | 4,096 | 63 optimizer steps | 111,096 | 13,376 |
| F1/G1 legacy | FP32 | 1 GPU, effective batch 16 | 1/1 | 512 | 500 optimizer steps | 1,082,879 | 1,082,369 |

The F4 and F8 target-sync settings both correspond to approximately one hard
sync per 500 environment transitions. The legacy F1 run differs in GPU count,
batch size, warmup size, total duration, and code generation, so it is useful as
a long-run reference but is not a clean F/G-only control.

## Quantitative trajectory

The following peaks are raw single-record maxima. The final values are averages
over the latest 1,000 optimizer records and are more representative of the
current state.

| Run | Peak loss (env step) | Peak Q (env step) | Peak raw grad norm (env step) | Final loss / TD | Final raw grad norm | Final Q / target Q |
|---|---:|---:|---:|---:|---:|---:|
| F4/G1 | 191.17 (70,296) | 1,478.20 (81,404) | 262.82 (98,356) | 18.50 / 18.95 | 55.44 | 396.55 / 380.52 |
| F8/G1 | 241.56 (60,992) | 1,699.15 (69,136) | 276.44 (77,336) | 61.24 / 61.73 | 109.31 | 1,143.56 / 1,097.48 |
| F1/G1 legacy | 1,061.49 (76,502) | 2,942.78 (74,518) | 868.02 (86,225) | 3.77 / 4.06 | 21.72 | 84.07 / 80.51 |

Additional observations:

- No NaN or infinity was found in any optimization JSONL snapshot. This supports
  the conclusion that BF16 was the immediate cause of the previous task
  interruption.
- FP32 does not remove the underlying value-scale excursion. It allows the run
  to survive the excursion and later recover.
- The raw gradient norm exceeded the configured clipping threshold of 10 in
  93.19% of F4 updates, 89.74% of F8 updates, and 99.28% of legacy F1 updates.
  `clip_grad_norm_` reports the norm before clipping, so the applied gradients
  remain bounded, but clipping is functioning as the normal update regime rather
  than an occasional safeguard.
- The latest mean Q-target bias is about +16.03 for F4, +46.08 for F8, and +3.57
  for F1. F4 and especially F8 are still materially overestimating their current
  bootstrap targets.
- At a comparable environment budget near 110k steps, F4 and F8 both have mean
  Q on the order of 1,100. Thus F/G changes the optimizer count and the precise
  peak, but it is not the root cause of the common value excursion.

## Does the rise-then-fall pattern need to be eliminated?

The curve does **not** need to be forced to become monotonic. In off-policy TD
learning, replay distribution drift, epsilon decay, a moving target network,
and bootstrapping can naturally make loss, TD error, gradient norm, and Q first
increase and later decrease. Training loss is not a supervised-learning
validation loss.

The current *magnitude*, however, still needs to be addressed. With at most 24
steps and normalized step reward usually near 0-1, the discounted upper bound
of the step-reward portion is approximately:

```text
(1 - 0.99^24) / (1 - 0.99) = 21.43
```

The terminal predictor can add values outside this simple step-only bound, but
the observed terminal reward scale does not explain Q values of 1,000-3,000.
Therefore the peak represents severe value miscalibration, even though FP32
eventually brings it down. It matters because the policy is ranked by these Q
values during the excursion, it makes mixed precision unsafe, and it wastes a
large amount of data and compute before recovery.

The metric decline alone is also not proof of policy improvement. Fixed-set
validation shows:

| Run | First validation mean | Best validation mean | Latest validation mean |
|---|---:|---:|---:|
| F4/G1 | 16.63 | 16.63 | 15.48 |
| F8/G1 | 17.49 | 17.49 | 16.05 |
| F1/G1 legacy | 15.82 | 17.38 | 16.30 |

Meanwhile the mean training episode return rose by roughly one point in all
three runs. This combination is consistent with fitting or exploiting the
shaped training reward without a demonstrated improvement on fixed validation
proteins. Selection should therefore be based on terminal mechanical-property
improvement and empirical-return calibration, not on loss decline.

## Most likely causes

1. **Positive shaping baseline.** Neutral H-bond terms are often 0.5 and the
   local-RMSD term is often close to 1. A mutation can receive a positive reward
   without producing a meaningful terminal mechanical improvement.
2. **Mutation reversal cycles.** All three configurations have
   `prevent_revisit_positions=false`. The live F8 tail contains an immediate
   M-to-W then W-to-M reversal at the same residue, receiving approximately 0.50
   and 0.79 reward. This gives the agent a repeatable positive local-reward loop.
3. **Time-limit state aliasing.** Replay correctly stores
   `done = terminated OR truncated`, and DDQN stops bootstrapping at max steps.
   However, remaining episode time is absent from the ESM-only observation, so
   the same sequence at different remaining horizons has different correct Q
   values but looks identical to the network.
4. **Large masked action maximum.** Each residue contributes 20 actions, often
   leaving thousands of valid candidates. Double DQN reduces but does not remove
   max-selection overestimation under noisy function approximation.
5. **Unnormalized Q head.** The per-residue head is currently Linear-ReLU-
   Linear-ReLU-Linear without input normalization or LayerNorm. Positive
   bootstrapped targets can drive activation and Q scale upward.
6. **Reward-frequency imbalance.** Equal weights inside the terminal predictor
   do not make terminal and step rewards equally influential: terminal reward is
   received once, while positive step shaping is received up to 24 times.

## Recommended stabilization sequence

The first objective is calibrated value learning, not cosmetically smaller
curves.

1. Add Q-versus-return diagnostics: record predicted Q and the realized
   discounted Monte Carlo return by remaining-horizon and protein-length bins.
   Add warning/stop thresholds based on empirical return scale.
2. Run an isolated `--prevent-revisit-positions` ablation. This removes both
   immediate reversals and longer same-position cycles without changing the
   optimizer.
3. Add normalized remaining horizon to the state/Q head, then rerun the same
   configuration. This restores the finite-horizon Markov information missing
   from the sequence embedding.
4. Center step shaping around zero or use potential differences, and scale the
   sum of step shaping by the episode horizon. A 0-1 component normalization is
   numerically bounded but creates a positive baseline; a centered range is more
   appropriate for shaping. Keep the terminal reward on a documented comparable
   return scale.
5. If Q is still overestimated, test `learning_rate=3e-5` and input LayerNorm as
   separate ablations. Then test a slower/soft target update and `gamma` in
   0.95-0.98. Do not change all of these at once if scientific attribution is
   important.
6. Do not use prioritized replay yet: repeatedly selecting high-TD transitions
   can amplify the current value-scale problem.

A healthy run does not require zero loss or raw gradient norm below 10 at every
step. Practical acceptance criteria are: no non-finite values, late-stage
clipping becomes occasional rather than nearly universal, Q agrees with
empirical finite-horizon returns, Q-target bias approaches zero, reversal rate
approaches zero, and fixed validation terminal strength/toughness beats both
wild-type and random-policy baselines.

## Why training is slow

Approximate end-to-end throughput, including validation/checkpoint pauses, is:

| Run | Approximate throughput |
|---|---:|
| F4/G1 | 1.32 environment steps/s |
| F8/G1 | 1.57 environment steps/s |
| F1/G1 legacy | 1.26 environment steps/s |

F8 is about 19% faster than F4 in this snapshot, but the bottleneck is not
primarily epsilon exploration. ESM2 observation encoding and PyRosetta local
repack still run for every environment transition, whether the action was
random or greedy. The current loop has one serial environment actor; the four
GPUs mainly parallelize the relatively small Q-head optimizer batch.

There are also severe I/O costs:

- F4 stdout is approximately 1.28 GB, F8 stdout 0.69 GB, and legacy F1 stdout
  8.33 GB because transition-level events are logged at INFO.
- F4/F8 replay checkpoints are approximately 47-49 GB. The legacy F1 run spent
  more than 20 minutes in one compressed replay save at the snapshot boundary.
- The replay stores both state and next-state per-residue ESM arrays, so adjacent
  transition embeddings are duplicated and expensive to compress.

Safe immediate speedups that do not change learning semantics are:

1. Move per-action, per-repack, per-buffer-add, and per-optimizer logs to DEBUG;
   retain periodic progress, JSONL summaries, warnings, and TensorBoard.
2. Save `agent.pt` frequently but save the full replay much less often. Use an
   atomic, chunked format and avoid `np.savez_compressed` on the critical path.
3. Add wall-time profiling for ESM encoding, PyRosetta repack, action selection,
   optimizer, validation, logging, and checkpointing before optimizing further.

The major architectural speedup is an actor-learner design: run multiple
PyRosetta environments in separate processes, batch their sequences for ESM2
encoding, and let one learner consume their transitions. This uses the four GPUs
for state collection/encoding instead of only splitting the small Q head. A
smaller ESM2 checkpoint or compact replay representation can be tested later,
but those choices affect representation quality and should be benchmarked.

## Recommended next work

1. Do not interrupt the active F4/F8 jobs solely because their curves are
   non-monotonic. Compare them at an equal 200k-250k environment-step boundary.
   The legacy F1 run has enough evidence; after its current replay save finishes,
   continuing it has low diagnostic value.
2. Treat F8/G1 FP32 as the current speed-oriented candidate, but do not declare
   it superior until its fixed validation terminal metrics and Q calibration are
   compared with F4 at equal environment steps.
3. Add the value-calibration and reversal diagnostics first. Then run sequential
   F8/G1 ablations: no-revisit, remaining-horizon state, and centered/horizon-
   scaled step shaping.
4. Once one configuration is calibrated, repeat it with at least three seeds.
   Select by terminal strength/toughness improvement over wild-type and random
   baselines, with wall-clock time as the secondary criterion.
5. In parallel, make the low-risk logging and replay-checkpoint I/O changes.
   Only after profiling should the environment collection path be converted to
   multi-process actors and batched ESM inference.

# 2026-08-06: reduced INFO logging and resumable sparse replay checkpoints

## Goal

This update addresses two independent sources of wall-clock overhead without
changing DDQN targets or reward semantics:

1. transition-level stdout logging;
2. serializing the full variable-length ESM2 replay at every agent checkpoint.

It also adds an episode-boundary restart path and an F4/G1 resume launcher.

## INFO log reduction

The following high-frequency messages were moved from INFO to DEBUG:

- action selection and decoded action;
- environment reset/load details for every episode;
- mutation, local-neighborhood, repack, minimization, and per-step completion;
- full step-reward metric dictionaries;
- ESM2 encoding for every sequence;
- ReplayBuffer add/sample operations;
- optimizer event start, warmup skip, raw optimizer completion, and record append;
- JSONL append and episode-CSV rewrite messages;
- candidate-PDB dump details.

INFO still reports information useful for normal monitoring:

- run configuration, dataset sizes, devices, ESM2/PyRosetta initialization;
- periodic progress according to `--log-every-steps`;
- one episode summary and one concise terminal-reward summary per episode;
- validation summary, target-network synchronization, warnings, and errors;
- agent/replay checkpoint start, completion, byte size, and elapsed time.

The detailed records are still written to JSONL and TensorBoard. Setting
`--log-level DEBUG` restores the verbose stdout trace for diagnosis.

## Why a 50,000-transition replay becomes 47-49 GB

The transition count is not unusually large for DDQN, but each replay item is
large because it stores both current and next per-residue ESM2 embeddings.
Ignoring small metadata, one transition needs approximately:

```text
state + next_state = 2 * L * 1280 * 4 bytes
two action masks   = 2 * L * 20 * 1 byte
total              = L * 10,280 bytes
```

The current F4 episode log contains 9,120 completed episodes at the analysis
snapshot, with mean sequence length 102.41 aa, median 89, and 90th percentile
177. At this mean length:

```text
raw bytes per transition ~= 1.053 MB
50,000 transitions       ~= 52.64 GB before compression
```

This agrees with the observed 47-49 GB archives. Replay capacity therefore
affects checkpoint size almost linearly:

| Capacity | Estimated raw embedding/mask data | Approximate episodes represented at 24 steps |
|---:|---:|---:|
| 50,000 | 52.6 GB | 2,083 |
| 20,000 | 21.1 GB | 833 |
| 10,000 | 10.5 GB | 417 |

Therefore, `replay_capacity=50000` is a direct cause of the archive size, but it
is not automatically "too large": reducing it also reduces protein diversity
and increases replay turnover. The full scripts keep 50,000 as the scientific
baseline and expose `MPRL_REPLAY_CAPACITY` for an isolated 20k-versus-50k
ablation. A reasonable first storage experiment is:

```bash
MPRL_REPLAY_CAPACITY=20000 bash train_update_frequency_f4.sh
```

Do not interpret that run as an F/G-only comparison, because capacity is a new
experimental factor.

## Checkpoint policy changes

`training.py` now separates cheap model checkpoints from expensive resumable
replay checkpoints:

- `--checkpoint-every N`: saves only `checkpoints/agent.pt`;
- `--replay-checkpoint-every N`: saves an episode-boundary resume bundle;
- `--no-save-final-replay`: skips a full replay save during finalization while
  still saving `agent_final.pt`.

The resume bundle is:

```text
checkpoints/resume/
├── agent.pt
├── replay_buffer.npz
└── training_state.json
```

`training_state.json` records the next episode, agent environment/optimization
steps, replay size/capacity/position, state/action dimensions, and deterministic
dataset schedule. Loading rejects inconsistent files instead of silently mixing
checkpoints from different save boundaries.

Agent, replay, and JSON writes now use temporary files followed by same-filesystem
atomic replacement. This prevents readers from seeing a partially written
archive. During replay overwrite, enough free space for the old archive and one
temporary new archive is required; with the 50k baseline, reserve roughly an
extra 50 GB.

The four-GPU launcher defaults are now:

```text
agent checkpoint:  every 240 episodes
resume replay:     every 2400 episodes
final replay:      disabled
```

Use `MPRL_SAVE_FINAL_REPLAY=1` when a final replay archive is explicitly needed.

## Resume behavior

On resume, the code restores:

- online and target networks;
- Adam state and AMP scaler state when applicable;
- epsilon/environment and optimizer counters;
- DDQN NumPy and PyTorch RNG states;
- ReplayBuffer content, ring position, and sampling RNG;
- next dataset episode by regenerating the seeded dataset order and skipping the
  already completed entries.

Resume occurs at an episode boundary. PyRosetta's process-global random state is
not checkpointed, so a resumed trajectory is state-consistent but not guaranteed
to be bitwise identical to an uninterrupted run.

### Preferred F4/G1 restart

For checkpoints generated by the new code:

```bash
bash resume_train_update_frequency_f4.sh \
  /path/to/f4-run/checkpoints/resume
```

The script starts a new output directory and retains F4/G1, batch 128,
`max_steps=24`, warmup 4096, and target sync 125. Starting a new output directory
avoids loading multi-gigabyte historical JSONL files into the logger.

### Legacy F4/G1 restart

Old run directories contain `agent.pt` and `replay_buffer.npz` but no
`training_state.json`. They require an explicit next episode:

```bash
bash resume_train_update_frequency_f4.sh \
  /path/to/old-f4-run/checkpoints \
  NEXT_EPISODE
```

Derive `NEXT_EPISODE` only from the last **completed** paired checkpoint:

```bash
rg 'Periodic checkpoint complete episode=' /path/to/train_stdout.log | tail -1
```

If the last completed record says `episode=8879`, use `NEXT_EPISODE=8880`.
Never load an old-format replay while its original job is still writing it;
wait for both `ReplayBuffer save complete` and `Periodic checkpoint complete`.

## Tests

Automated tests in the `mprl-vgpt` environment:

```text
targeted checkpoint/replay/logger suite: 57 passed, 2 skipped
complete project suite:                  112 passed, 2 skipped
```

An actual two-process PyRosetta smoke test used F4/G1 scheduling with a tiny
one-protein, one-step CPU configuration:

```text
first run:  next_episode=1, environment_steps=1, replay_size=1
resumed run: episode=1 completed, next_episode=2,
             environment_steps=2, replay_size=2
```

Artifacts are in:

```text
/tmp/mprl-resume-smoke-rtYzXM
```

The resumed INFO output contained no per-action, per-ESM, per-replay-sample, or
per-optimizer trace. The smoke validates checkpoint consistency and training-loop
continuation; it intentionally does not load the active 47-49 GB F4 replay or
start another four-GPU job.

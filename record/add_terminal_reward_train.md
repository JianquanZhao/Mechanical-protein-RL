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

# Asynchronous actor-learner implementation (2026-08-07)

## Motivation and design overview

The previous `multi` mode used `torch.nn.DataParallel` only for replay-batch
forward/backward passes. Protein loading, PyRosetta mutation/repack, reward
calculation, and per-step ESM2 encoding still ran in one serial environment.
The new `asynchronous` branch introduces an actor-learner pipeline that attacks
the collection bottleneck directly:

```text
CPU PyRosetta actor processes
        | sequence requests                 | transitions
        v                                   v
GPU 1-3 batched ESM2 workers        bounded event queue
                                                |
                                                v
                                  GPU 0 DDQN learner + replay
                                                |
                                     periodic policy snapshots
                                                |
                                                v
                                      CPU actor Q heads
```

The implementation uses `spawn`, not threads or CUDA-after-fork. Every actor
owns its environment, Pose, reward calculators, RNG, and lightweight CPU Q
head. PyRosetta Pose objects never cross a process boundary. Queues carry only
sequences, NumPy embeddings, masks, rewards, episode metadata, and plain CPU
network state dictionaries.

## Detailed implementation

### Batched ESM2 inference

`model/encoding_module/esm2_encoder.py` now provides:

- `encode_sequence(sequence)` for the compatible single-sequence path;
- `encode_sequences(sequences)` for padded variable-length batches;
- one ESM2 forward call per batch, followed by per-sequence removal of BOS,
  EOS, and padding tokens;
- `torch.inference_mode()` for inference-only execution.

`model/asynchronous_module/runtime.py` adds a shared inference request queue,
one response queue per actor, request IDs, timeout/error propagation, and
dynamic batching. One worker is created per configured inference GPU. A worker
waits for the first request, collects up to `--async-inference-batch-size`
requests during `--async-inference-batch-wait-ms`, then executes one model call.
The bounded queues provide backpressure when actors outrun inference or the
learner.

### Parallel PyRosetta actors

Each actor process:

1. creates an independent `MechanicalProteinEnv` and PyRosetta backend;
2. obtains reset/step observations through `RemoteESM2Encoder`;
3. selects masked epsilon-greedy actions with a CPU copy of the residue-wise Q
   head;
4. sends complete transition arrays to the central learner;
5. sends episode summaries separately and optionally writes its own candidate
   PDB;
6. consumes only the latest available learner policy snapshot.

The actor-side epsilon estimate advances by the actor count per local step so
its decay approximates the global transition schedule between weight syncs.
Every policy snapshot includes the authoritative global environment-step count,
which corrects this estimate periodically. Actor seeds are separated by a
deterministic offset.

### Central learner

`asynchronous_training.py` owns the only replay buffer and optimizer. It:

- writes transitions to the existing variable-length padded replay buffer;
- defines `environment_steps` as transitions accepted by the learner;
- preserves global `--train-frequency F` / `--gradient-steps G` semantics;
- runs the DDQN learner exclusively on `--device` (the production script uses
  `cuda:0`);
- broadcasts online-network CPU snapshots every
  `--async-policy-sync-interval` optimizer steps;
- tracks actor ID and policy version in step/episode records;
- reports collection throughput and propagates all worker failures to the
  parent process instead of waiting indefinitely.

The root `training.py` parser now accepts `--mode asynchronous` and dispatches
to the new learner. GPU overlap is rejected: the learner device and ESM2 worker
GPU IDs must be disjoint.

### Logging scalability

The logger previously rewrote the complete episode CSV after every episode and
retained every transition and optimizer record in memory. That behavior becomes
prohibitively expensive for hundreds of thousands of episodes. New options are:

```text
--episode-csv-every 1000
--max-step-records-in-memory 100000
--max-optimization-records-in-memory 100000
```

JSONL remains the complete append-only source of truth. The limits only affect
the recent in-memory window used to render plots. `episodes.csv` is refreshed
periodically and once more during clean logger shutdown. Existing direct logger
users retain the old defaults unless they select the new options.

## Launch scripts

The production launcher is `train_version_terminal_async.sh`. Its default GPU
layout is:

```text
learner:             cuda:0
ESM2 workers:        cuda:1,cuda:2,cuda:3
PyRosetta actors:    12 CPU processes
ESM2 batch size:     8 per worker
update schedule:     F8/G1, replay batch 128
```

Start it with:

```bash
bash train_version_terminal_async.sh
```

The main tuning controls can be overridden without editing the script:

```bash
MPRL_ASYNC_ACTORS=8 \
MPRL_ASYNC_INFERENCE_BATCH_SIZE=8 \
MPRL_TRAIN_FREQUENCY=8 \
bash train_version_terminal_async.sh
```

`train_asynchronous_smoke.sh` is the foreground two-GPU diagnostic launcher.
It uses GPU 0 for the learner, GPU 1 for ESM2, two actors, four episodes, and
two steps per episode.

## Tests and observed results

Automated tests cover variable-length ESM batching, remote request/response
matching, dynamic worker batching, variable-length masked actor actions, and
all pre-existing project behavior:

```text
targeted asynchronous/DDQN/replay/multi-GPU tests: 72 passed, 2 skipped
complete project test suite:                       115 passed, 3 skipped
bash syntax, Python compilation, git diff check:   passed
```

The real smoke test used the `mprl-vgpt` environment, two RTX 4090 GPUs, local
ESM2 parameters, and two independent PyRosetta processes. No `nvidia-smi`
decision was used; availability was established by the PyTorch CUDA runtime.

```text
episodes:                 4/4
transitions:              8
optimizer updates:        2
final checkpoint:         written
diagnostic figures:       10
process exit code:        0
total wall time:          41.083 s (includes ESM2/PyRosetta startup and plots)
first actor episodes:     6.651 s and 6.822 s (cold start)
later actor episodes:     0.850 s and 1.210 s
```

Artifacts are in:

```text
/tmp/mprl-async-smoke-20260807-codex-small
```

The first smoke attempt was manually stopped during validation of all 7701 PDB
index entries, before workers started. The successful smoke used four training
and two validation entries so it measured the asynchronous path rather than
full dataset preprocessing.

This smoke proves process isolation, remote ESM2 inference, replay ingestion,
learner optimization, logging, checkpointing, and clean shutdown. It is not a
throughput benchmark for the terminal-reward production workload. Actor counts
8, 12, and 16 should be benchmarked with identical environment-step budgets;
the best setting is the first one where ESM GPU utilization or learner queue
latency saturates rather than the largest process count.

## Current constraints

- Exact replay resume is disabled in asynchronous mode. Completion order can
  differ from episode ID, so a correct checkpoint must also atomically record
  assigned and in-flight tasks. Agent-only periodic checkpoints are supported.
- Periodic in-process validation is deferred. Blocking the learner for serial
  validation can fill transition queues and distort collection throughput;
  saved checkpoints should currently be evaluated by a separate process.
- Every actor loads its own terminal random-forest artifact. Monitor host RAM
  before increasing actor count substantially.
- Three ESM2 model replicas require enough memory on GPUs 1-3. Lower
  `MPRL_ASYNC_INFERENCE_GPUS` or the worker count if a selected checkpoint is
  too large.

## Branch status

The local `asynchronous` branch was created from
`feature/mechanical-property-predictor`. The initial remote push was blocked by
the execution security review because repository visibility and outbound code
scope could not be verified automatically. No workaround was attempted; remote
synchronization remains pending explicit approval after this implementation is
reviewed.

# 异步与串行 F8/G1 训练对比分析（2026-08-07）

## 分析对象与口径

本次比较使用：

```text
异步日志：async_a12_esm8_f8_g1_20260807_111457/train_stdout.log
串行日志：update_frequency_f8_g1_bs128_h24_20260806_155311/train_stdout.log
```

两次训练的主要共同参数是 batch size 128、F8/G1、replay warmup 4096、
replay capacity 50000、`max_steps=24`、学习率 `1e-4`、无 AMP。为了避免
冷启动和 replay warmup 影响，速度主比较区间选择两边共同经历的
`global_step=50000 -> 99000`，共 49000 个 transition。稳定性比较选择
两边共同具有的前 12023 次 optimizer update。

该实验并不是完全严格的单变量对照，至少存在三个重要差异：

1. 串行版 `target_sync_interval=63`，异步版为 125；
2. 串行版每 240 episodes 运行 16 个 validation episodes，异步版将周期
   validation 延后到独立 checkpoint 评估；
3. 串行版每 2400 episodes 保存约 47-49 GB replay，异步版关闭周期 replay
   checkpoint。

因此下面同时报告原始端到端速度和扣除这些维护任务后的训练流水线速度。

## 速度量化结果

### 原始端到端吞吐

在 `50k -> 99k transitions` 区间：

| 版本 | 开始时间 | 结束时间 | 用时 | 吞吐 | 折算 episodes/hour |
|---|---:|---:|---:|---:|---:|
| 异步 A12 | 13:07:19 | 14:42:48 | 5729 s | 8.55 step/s | 1283 |
| 串行 | 19:34:47 | 00:05:01 | 16214 s | 3.02 step/s | 453 |

原始端到端结果为：

```text
吞吐倍数：8.55 / 3.02 = 2.83x
速度提升：(2.83 - 1) * 100% = 183%
完成同样 49000 transitions 的墙钟时间减少：64.7%
```

如果该稳态吞吐始终保持不变，完成 492864 个 24-step episodes 的粗略
外推约为异步 16.0 天、串行 45.3 天。该数字只能用于容量规划，不能替代
完整训练实测，因为后续 checkpoint、绘图、文件增长和硬件负载会变化。

### 扣除非等价维护任务

相同区间内，串行版发生了：

```text
9 次绘图 + 9 次 16-episode validation：553 s
1 次 49 GB replay checkpoint：             2722.942 s
维护总时间：                               3275.942 s
```

异步版发生 9 次绘图，共 78 s，没有周期 validation 和 replay checkpoint。
扣除这些不等价任务后：

| 版本 | 估计有效训练时间 | 有效吞吐 |
|---|---:|---:|
| 异步 A12 | 5651 s | 8.67 step/s |
| 串行 | 12938.058 s | 3.79 step/s |

较公平的训练流水线提升为：

```text
有效吞吐倍数：8.67 / 3.79 = 2.29x
有效速度提升：129%
有效训练时间减少：56.3%
```

在原始时间差 10485 s 中，约 3198 s（30.5%）来自 validation、绘图和
replay checkpoint 配置差异；约 7287 s（69.5%）来自异步采样和批量 ESM2
流水线本身。因此“速度约提升 1 倍”的判断是合理且偏保守的；严格按相同
训练工作量，当前证据支持约 2.3 倍吞吐，而不是 12 倍。

## 是否符合理论提升

### 与朴素 12 倍上限的差距

12 个 actor 的 12 倍只是所有工作都能独立并行、没有共享瓶颈和资源竞争
时的上限，当前实现显然不满足这些条件。稳态区间中：

```text
串行普通 episode 平均耗时（排除每 240 次维护 episode）：6.338 s
异步单 actor episode 平均耗时：                         22.509 s
单个异步 actor 相比串行变慢：                           3.55x
12 actors 按实测单 actor 服务时间计算的容量上限：        12.80 step/s
异步实际有效吞吐：                                      8.67 step/s
并行流水线利用率：                                      67.7%
```

CPU/GPU 竞争已经把理论相对上限从朴素的 12 倍压缩到约
`12 / 3.55 = 3.38x`；实际获得 2.29 倍，相当于该实测容量上限的约 67.7%。
对于第一版包含 PyRosetta、ESM2、IPC 和 learner 的异步流水线，这个结果
合理，但仍有明确优化空间。

### 未达到更高加速比的原因

1. **PyRosetta CPU 与内存竞争**：12 个进程同时进行 PDB 清洗、Pose 构建、
   mutation 和 repack，争用 CPU core、内存带宽、文件系统和临时文件目录。

2. **ESM2 实际 batch 难以达到 8**：只有 12 个同步等待响应的 actor，却有
   3 个 ESM worker。均匀分配时每个 worker 同时只能看到约 4 个请求，配置的
   batch size 8 多数时候只是上限，并不代表实际 batch 为 8。

3. **随机长度 padding 浪费**：不同长度蛋白被放入同一 ESM batch，整个
   batch 按最长序列 padding。一个长蛋白会显著增加同 batch 中短蛋白的计算量。

4. **大数组跨进程复制**：每个 `L x 1280` float32 embedding 先从 ESM worker
   回到 actor，再随 state/next_state 发送到 learner；相邻 transition 还会重复
   发送前一步的 next_state，产生序列化、内存复制和 IPC 压力。

5. **actor 是同步 RPC 客户端**：每个 actor 每一步都等待 ESM 返回，不能在
   同一 actor 内将结构计算、下一请求和当前 embedding 传输重叠起来。

6. **共享 learner 和日志主循环**：GPU 0 learner、replay padding/sample、JSONL、
   TensorBoard、绘图和 episode task 分发都在主进程。绘图期间主进程不能及时
   排空 event queue，也不能马上给空闲 actor 分发下一任务。

7. **蛋白长度和结构复杂度造成 straggler**：异步 episode 的 p95 耗时约
   35.1 s，均值 23.9 s。慢结构会占用 actor slot，并降低 12 个 actor 的整体利用率。

## Loss、Q value 和 TD error 峰值量化

以前 12023 次 optimizer update 为共同区间：

| 指标 | 异步最大值 | 串行最大值 | 异步/串行 | 串行/异步 |
|---|---:|---:|---:|---:|
| loss | 18.085 | 274.089 | 6.60% | 15.16x |
| mean absolute TD error | 18.557 | 274.588 | 6.76% | 14.80x |
| mean Q value | 161.007 | 1880.970 | 8.56% | 11.68x |
| grad norm | 66.189 | 261.510 | 25.31% | 3.95x |

最近 500 次共同 update 的均值更贴近 TensorBoard 中观察到的“约 1/10”：

| 指标 | 异步均值 | 串行均值 | 异步/串行 |
|---|---:|---:|---:|
| loss | 7.789 | 75.670 | 10.3% |
| mean absolute TD error | 8.209 | 76.162 | 10.8% |
| mean Q value | 155.297 | 1417.813 | 11.0% |

该差异不是由异步版获得了更小的 reward 造成的。前 99000 transitions 的
reward 统计几乎一致：

```text
异步 step reward：  mean=0.6451, std=0.1791
串行 step reward：  mean=0.6424, std=0.1796
异步 episode return：mean=15.4839, std=1.0856
串行 episode return：mean=15.4179, std=1.1342
```

因此 Q/TD 峰值下降主要来自训练动力学和样本组织方式，而不是 reward scale。

## 峰值下降的机制分析

### 1. transition 时间相关性被大幅削弱

串行版连续写入同一个 24-step episode，前 99000 transitions 中相邻记录仍
属于同一 episode 的比例为 95.83%，连续 run 的中位数是 24。异步版来自
12 个 actor 交错写入，相邻记录属于同一 episode 的比例只有 0.0081%，连续
run 中位数为 1、最大值为 2。

这使 replay 在训练早期就同时包含多种蛋白、位置和突变轨迹，降低连续相似
状态造成的梯度同向累积。虽然 replay buffer 满后 uniform sampling 本身也会
打乱顺序，但在线训练过程看到的“当时 buffer 内容”仍更丰富，尤其是在最容易
出现 bootstrap 正反馈的 update 1000-5000 区间。该区间串行/异步峰值比分别
达到 loss 19.59 倍、TD error 18.64 倍、Q value 15.40 倍。

### 2. actor policy lag 切断即时正反馈

异步 actor 每 100 optimizer steps 接收一次 learner 权重。日志中 actor 相对
learner 的平均滞后约 58 optimizer updates，即约 464 transitions；p95 滞后约
103 updates，即约 824 transitions。

串行版中 online Q 一旦高估某些 action，这些 action 会立即影响下一步 greedy
采样，新 transition 又会反过来强化相同 Q 估计。异步版的旧策略行为相当于一个
低通滤波器，learner 的短期异常不会马上同步到所有 actor。DDQN 是 off-policy
算法，可以使用这些较旧策略产生的数据，因此适度 policy lag 在这里起到了稳定
作用。过大的 lag 仍可能损害最终策略质量，不能把“越旧越稳定”等同于“越好”。

### 3. target network 更新频率降低

共同 12023 updates 内：

```text
异步 target sync：96 次，interval=125
串行 target sync：190 次，interval=63
```

串行版大约每 504 transitions 将 online network 的高估复制到 target；异步版
约每 1000 transitions 才复制一次。较慢 target sync 延缓了
`online overestimate -> target increase -> TD target increase -> online increase`
的 bootstrap 正反馈，这很可能是峰值下降的重要原因之一。

这也是当前实验最重要的混杂变量。没有在相同 target interval 下重跑之前，不能
把 10 倍稳定性改善全部归因于异步 actor。

### 4. 多 actor 的独立随机轨迹提高覆盖度

12 个 actor 使用不同 RNG seed，并在不同蛋白上同步探索。即使全局 epsilon
相同，随机 action、结构更新耗时和完成顺序也不同。较高的状态动作覆盖度降低了
单一轨迹中偶然高 Q action 被反复强化的概率。

### 5. Huber loss 与 TD error 本来就会同步变化

当前 `huber_beta=1`。当绝对 TD error 远大于 1 时，Smooth L1 loss 近似
`abs(TD error) - 0.5`。因此 loss 和 mean absolute TD error 同时下降约 15 倍
并不是两个独立机制，而是 TD target 与 Q 差距下降在两个统计量上的一致表现。

## 仍然存在的稳定性问题

异步版是显著改善，不是彻底解决。共同区间内异步 mean Q 峰值仍为 161，最近
500 updates 的均值仍约 155，而实际未折扣 episode return 均值只有约 15.48；
折扣 return 还会更低。也就是说异步 Q 仍大约高估一个数量级，只是串行版最近
均值约 1418 的近百倍高估被强烈抑制了。

异步日志末端的 recent-500 指标为：

```text
loss=7.789, TD error=8.209, grad norm=24.913, mean Q=155.297
```

其中 grad norm 仍经常高于 clip threshold 10，Q value 也仍处于高平台。后续
判断应继续依赖固定验证集实际 return、terminal strength/toughness 改善和
Q-versus-Monte-Carlo-return calibration，而不能只凭 loss 峰值较低就判断策略
已经收敛。

## 结论与下一组必要对照

1. 异步版本原始端到端吞吐为串行版 2.83 倍；去除不等价 validation、绘图和
   replay checkpoint 后，核心流水线约为 2.29 倍。速度提升真实且主要来自架构，
   但并未达到 12 actors 的朴素 12 倍理论上限。

2. 当前 2.29 倍约达到依据实测 actor 服务时间推导容量上限的 67.7%。下一步
   提速应优先记录 ESM 实际 batch size、request wait、actor PyRosetta time、IPC
   time 和 learner duty cycle，并尝试按序列长度 bucket batching。

3. loss、TD error、Q value 下降到约 1/10 的观察成立。主要解释是 transition
   去相关、独立 actor 探索和 policy lag 抑制即时反馈，同时 target sync 从 63
   改为 125 也显著减慢高估传播。

4. 最小因果对照应固定相同数据、步数、validation/checkpoint 配置并依次比较：
   `serial-target125`、`async-target125-policy-sync100`、
   `async-target63-policy-sync100`、`async-target125-policy-sync1`。只有这样才能
   分离异步采样、target sync 和 policy staleness 各自对稳定性的贡献。

# 2026-08-10：三日异步训练的 actor 扩容与 total reward 分析

## 分析范围

本次分析读取：

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
async_a12_esm8_f8_g1_20260807_111457/
```

量化快照包含 `98,611 / 492,864` 个 episode、`2.36M+` transitions 和
`295k+` optimizer updates，约完成总计划的 20%。配置为 12 个 PyRosetta
actor、3 个 ESM2 inference worker、F8/G1、batch size 128、24 steps/episode、
`epsilon_end=0.05`、无 AMP。该任务在分析期间仍在写日志；本次没有停止进程，
也没有修改训练代码。

## 是否应停止 A12 并增加 actor

停止当前三日 A12 任务、保留其作为诊断基线，然后测试更多 actor，方向合理：

1. A12 已跨越 12 个完整 epoch，足以确认吞吐、reward 平台和数值行为，不必再用
   13 天仅验证同一个平台是否延续。
2. 当前平均吞吐约 `8.336 step/s`，约 `1,250 episode/hour`，外推完整 64 epoch
   总耗时约 16.4 天，与 15–17 天的观察一致。
3. 单 actor episode 平均耗时 `23.05 s`，12 actor 按实测服务时间计算的容量约为
   `12 * 24 / 23.05 = 12.50 step/s`；实际 8.336 step/s，相当于约 66.7% 的
   actor-side 容量利用率。增加并发有机会隐藏 ESM2/IPC/learner 等待并提高 batch
   填充率，但当前仍有约三分之一损失来自共享流水线，不能期待 actor 数量线性加速。
4. 服务器有 56 个物理核、112 个逻辑 CPU 和约 79 GiB available memory，A18/A24
   在 CPU 数量上可行；但 swap 已使用约 `7.9/8.0 GiB`，继续扩容前必须监测每个
   PyRosetta actor 的 RSS、内存带宽和 NUMA 竞争。

不建议立即启动更大 actor 数量的 64-epoch 完整训练。推荐依次运行 A12、A18、A24
的相同短基准，每组至少经过 replay warmup，并比较固定 20k–50k transitions 区间。
三张 ESM GPU、每 worker batch size 8 时，A24 是第一个值得重点测试的配置，因为
理论上可为每张 GPU 提供约 8 个并发请求。只有 A24 的 ESM 实际 batch size、队列
等待、actor p95、learner duty cycle 和内存均健康时，才继续测试 A32/A36。

当前异步模式不保存可精确恢复的 replay/in-flight task 状态，因此停止后不能从
现有 agent checkpoint 精确续跑同一随机过程。最新周期 checkpoint 可保留用于离线
评估，但 actor 数量对比和下一次完整训练应从相同初始权重、seed 和空 replay 开始，
否则吞吐与策略效果会混入 warm-start 差异。当前 checkpoints 与 logs 分别约为
2.5 GiB 和 3.9 GiB，也应适当降低 plot/checkpoint/CSV 刷新频率后再做扩容测试。

## 当前 total reward 的精确定义

当前 `total_reward` **不是**“episode 终态力学性能减去初态力学性能”。异步 actor
在 `model/asynchronous_module/runtime.py` 中执行：

```text
episode_total_reward = sum(reward_t for t in 1..T)
reward_t = step_reward_t                         (非终止步)
reward_T = step_reward_T + terminal_reward_T    (最后一步)
```

因此本次固定 24-step episode 的记录是：

```text
total_reward = 24 个 normalized step reward 的和 + 1 个 terminal reward
```

step reward 虽然由当前结构相对 previous/reference 的碰撞、氢键和局部 RMSD 构成，
但经过非线性 0–1 映射并带有正基线，不能像势函数差那样在 24 步后相消为
`final - initial`。terminal reward 也是终态预测 strength/toughness 的绝对 z-score
平均，而不是相对初始结构的增量。

此外，当前 dual-structure wrapper 根据 **source PDB stem** 查找预先生成的预测结构。
命中时，该 predicted PDB 对同一 source protein 的所有 mutation episode 都不变，
并不是根据本 episode 的 terminal mutated sequence 重新预测的结构；其 50% 分量会
稀释终态突变信号。未命中时才只使用 PyRosetta terminal packed structure。

## total reward 趋势的量化结果

全部 98,611 个 episode：

| 指标 | 均值 | 标准差 | 与 total reward 的相关性 |
|---|---:|---:|---:|
| total reward | 15.9659 | 1.1421 | 1.000 |
| 24-step reward sum | 16.0442 | 1.1849 | 0.897 |
| terminal reward | -0.0783 | 0.5293 | 0.149 |

terminal reward 均值只占 total reward 均值的约 `-0.49%`，total reward 主要由
24 个正值 step reward 决定。按完整 epoch 比较：

| Epoch | Episodes | Total | Step sum | Terminal | Mean epsilon |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,701 | 15.7082 | 15.7929 | -0.0847 | 0.1756 |
| 1 | 7,701 | 15.9753 | 16.0528 | -0.0776 | 0.0500 |
| 5 | 7,701 | 15.9578 | 16.0372 | -0.0794 | 0.0500 |
| 8 | 7,701 | 16.0077 | 16.0864 | -0.0787 | 0.0500 |
| 11 | 7,701 | 16.0186 | 16.0974 | -0.0788 | 0.0500 |

epoch 0 到 epoch 1 的 `+0.2671` 主要与 epsilon 从探索阶段降至 0.05 同时发生。
排除该阶段后，同一批 7,701 个 PDB 在 epoch 1 到 epoch 11 的配对结果为：

```text
total reward mean delta:   +0.0434（约 +0.27%）
step reward sum delta:     +0.0446
terminal reward delta:     -0.0012
reward improved fraction:  50.81%
```

改善比例接近 50%，说明 plateau 后没有可检测的、跨蛋白一致的策略提升。episode
completion order 与 total reward 的 Pearson 相关仅 `0.0428`，terminal reward 与
顺序的相关仅 `0.0032`。图中最初的上升是真实的 step-policy 改善，但后续基本持平。

## 为什么 loss/TD error 回落但 total reward 不增长

### 1. TD 收敛不等于策略改善

DDQN loss 衡量 Q 与 bootstrap target 是否一致，不直接衡量 terminal mechanical
property。online/target network 可以收敛到彼此一致但整体高估的固定点。最新 5,000
次 update 均值仍为：

```text
loss=3.492, mean_abs_td=3.746, mean_q=81.73,
mean_target_q=78.39, grad_norm=8.87
```

而 episode total reward 均值只有约 15.97；按 `gamma=0.99` 折扣后 return 还更低。
Q 仍显著高估，所以“从峰值回落”应称为数值稳定化，尚不能称为价值校准完成。

### 2. 正基线 step reward 淹没 terminal 信号

中性氢键变化被映射为 0.5；不增加 collision loss 时碰撞项为 1；相邻结构 RMSD
很小时 RMSD 项接近 1。一个几乎不改变结构的动作可以得到约 `0.7857` 的 step
reward。固定 24 步会累积约 16 分，而 terminal reward 只在最后出现一次、均值约
-0.078。优化总回报最容易的方向因此是“每一步少破坏”，不是提高终态 strength
或 toughness。

首尾各 10,000 transitions 的比较进一步显示：

```text
step reward:                 0.6174 -> 0.6750
collision unit score:        0.4573 -> 0.6337
backbone hbond unit score:   0.4917 -> 0.4990
sidechain hbond unit score:  0.4765 -> 0.4984
local RMSD unit score:       0.9905 -> 0.9961
```

主要进步来自减少“相对上一步”的 collision 恶化，氢键项只是回到中性基线，不能
证明力学性能提高。

### 3. step reward 存在局部安全但全局不优的漏洞

当前默认 `collision_penalty_mode="delta"`、`rmsd_penalty_mode="previous"`。
只要本步 collision 不比上一步更差，collision 项就可得到满分，即使当前结构已经
远差于 episode 初始结构。日志中一个终止 transition 的 collision score 为 81.60、
reference 为 11.55，但由于它比 previous 的 87.64 有所下降，collision reward 仍为
1.0。最后 10,000 steps 的 current collision 相对 reference 平均约 6.60 倍，
`collision_excess_over_reference > 10` 的比例仍为 74.59%。

这会鼓励“先恶化、再局部恢复”或反复突变，而不是保持全局结构质量。当前
`prevent_revisit_positions=false`，重复位置修改与反向突变进一步放大该问题。

### 4. training reward 混合不同蛋白，且没有固定验证曲线

每个 episode 从不同 CATH PDB 开始，不同蛋白的基线、长度、可突变位置和预测器
难度不同。训练 total reward 是 epsilon-greedy、stale actor policy 下的 on-run
统计，不是同一初态上的 deterministic evaluation。本次配置为
`validate_every=0, validation_episodes=0`，所以现有曲线本来就不是适合判断策略
改善的指标。正确主指标应是固定 validation PDB、固定初始结构、`epsilon=0` 下的：

```text
delta terminal strength, delta terminal toughness,
delta terminal reward, success/top-k rate, Q-versus-realized-return calibration
```

### 5. observation 与 reward 仍存在部分可观测性

ESM2 observation 主要编码当前序列，但 reward 依赖 PyRosetta 当前 Pose、相对上一步
结构、初始 reference、已访问位置和剩余 horizon。网络不能仅凭序列恢复这些变量，
相同或相似序列可能对应不同结构历史和不同 TD target，限制了 Q 与策略的可学习性。

## 推荐的下一阶段顺序

1. **先做吞吐 benchmark，不直接做完整训练**：A12/A18/A24 各跑相同 20k–50k
   transitions，关闭高频绘图，记录 ESM 实际 batch size、inference wait、PyRosetta
   time、IPC time、event queue depth、policy lag、RSS 和吞吐；按 step 数而非运行时间
   比较。
2. **建立固定验证器**：每个 checkpoint 在相同 validation PDB 上用 epsilon=0 评估，
   同时保存 initial/final strength、toughness 和结构质量。没有该曲线，不应启动
   15 天完整训练。
3. **把优化目标改为 improvement**：terminal reward 使用
   `z(property_final) - z(property_initial)`；在不能为 terminal mutated sequence
   实时预测结构前，不把静态 source predicted PDB 当作 50% 的终态评分。
4. **降低 step reward 的生存基线**：将 step reward 零中心化并按 horizon 缩放，
   collision 同时约束 `delta` 与 `excess_over_reference`，RMSD 同时保留 previous 与
   reference 约束，并启用禁止重复位置修改。这样 terminal mechanical signal 才能
   在 episode return 中占有可辨认的比例。
5. **补足状态**：至少加入 remaining-step fraction、visited-position mask 和必要的
   structure/reward summary，降低非 Markov target 噪声。

总体判断：停止当前 A12 诊断任务并进入 actor scaling 合理，但“更多 actor”只能解决
墙钟时间，不能解决 reward plateau。下一次完整训练的启动条件应同时包括：A24 等配置
达到吞吐饱和点、固定验证曲线可用、terminal reward 改为相对初态的力学性能改善信号。

# 2026-08-12：A36 训练效果诊断与下一阶段重点

## 本次判断

A36 已经足以回答 actor scaling 问题：继续增加 actor 的边际收益很低，不再深入做
系统级加速是合理决策。当前真正的瓶颈不是“收集的数据不够快”，而是 **训练目标、
观测状态和评价方法尚未对齐到终态力学性能改善**。如果保持现有 reward 继续完成
64 epochs，最可能得到的是一个更充分拟合“局部少破坏”信号的策略，而不是可靠提高
strength/toughness 的策略。

本次快照读取：

```text
A36: async_a36_esm8_f8_g1_20260811_145618
A12: async_a12_esm8_f8_g1_20260807_111457
```

A36 分析时包含约 36.4k episodes、874k transitions 和 108.8k optimizer updates。
两组训练的 F8/G1、batch size 128、24-step horizon、reward scale、target sync 和
epsilon schedule 相同，因此可在相同 episode/transition 预算下直接比较。

## A36 的速度结论

| 指标 | A12 | A36 | 变化 |
|---|---:|---:|---:|
| 全局吞吐 | 8.336 step/s | 9.596 step/s | +15.1% |
| 平均 actor episode 时间 | 23.55 s | 78.44 s | 3.33x 更慢 |
| 24 h 完成量 | 约 30k | 35,489 | 小幅增加 |
| 64 epochs 外推 | 约 15–17 天 | 约 13 天 | 非数量级改善 |

actor 数量增加 3 倍，但吞吐只提升约 15%。单 actor 变慢 3.33 倍，说明 CPU/NUMA、
内存带宽、PyRosetta、ESM2 请求、IPC 和主进程事件处理已经进入共享资源竞争区间。
A36 的 actor policy lag 也由 A12 的平均 62.3 updates 增至 67.1 updates。继续增加
actor 不太可能改变项目周期，因此后续只需保留 A12/A36 结果作为系统容量依据。

## 当前训练效果存在的问题

### 1. terminal mechanical reward 没有改善

相同约 36.4k episodes 下：

| 指标 | A12 | A36 |
|---|---:|---:|
| total reward mean | 15.9218 | 15.9039 |
| 24-step reward sum mean | 16.0013 | 15.9850 |
| terminal reward mean | -0.0795 | -0.0811 |
| terminal reward/order correlation | 0.0056 | 0.0052 |

A36 没有比 A12 获得更高的 total reward 或 terminal reward。A36 在 epsilon 已降到
0.05 后，epoch 1 到 epoch 3 的同一 PDB 配对结果为：

```text
total reward delta:       +0.00086
step reward sum delta:    +0.00152
terminal reward delta:    -0.00066
improved fraction:        50.69%
```

该变化远小于 episode 标准差约 1.08，改善比例接近随机的 50%。部分 epoch 4 相对
epoch 1 的 total reward 反而下降约 0.059。结论不是“增长较慢”，而是当前证据中
**不存在跨蛋白一致的 terminal mechanical improvement**。

### 2. loss/TD 回落没有形成可靠的价值函数

A36 的峰值相对 A12 略低：Q value peak 141.09 vs 161.13，loss peak 15.59 vs
18.08；但相同预算下最近 5,000 updates 为：

| 指标 | A12 | A36 |
|---|---:|---:|
| loss | 3.5088 | 3.3441 |
| mean absolute TD error | 3.7813 | 3.6123 |
| mean Q | 82.26 | 78.35 |
| mean target Q | 78.94 | 75.18 |
| grad norm | 10.75 | 10.05 |

数值略有改善，但 mean Q 仍远高于实际未折扣 episode return 约 15.9，且一个随机
transition 的真实剩余 return 通常比完整 episode return 更低。online Q 与 target Q
可以一起收敛到高估固定点，因此 loss/TD error 从峰值下降只能证明 Bellman 自洽性
有所恢复，不能证明 action ranking 正确或策略变好。

### 3. agent 主要优化局部 step reward，而不是力学性能

A36 首尾各 10k transitions 的变化为：

```text
mean step reward:              0.6151 -> 0.6683
collision unit score:          0.4489 -> 0.6036
backbone-H-bond unit score:    0.4912 -> 0.4984
sidechain-H-bond unit score:   0.4748 -> 0.4935
local-RMSD unit score:         0.9901 -> 0.9957
terminal contribution/step:  -0.00419 -> -0.00149
```

可观察到的提升主要来自 collision 项和本来就接近满分的 previous-step RMSD 项。
H-bond 项只是趋近无变化对应的 0.5 中性基线，terminal signal 基本没有变化。
24 个正值 step reward 累积约 16 分，而 terminal reward 只出现一次、均值约 -0.08，
因此最容易学习的策略是“每一步获得安全分”，不是提高终态预测力学性能。

### 4. reward 可以被局部恢复和循环动作利用

当前默认 collision 只惩罚相对 previous pose 的恶化，RMSD 也主要比较 previous
pose。A36 后 10k transitions 中：

```text
collision_excess_over_reference mean = 476.10
collision_excess_over_reference > 10 = 77.56%
```

也就是说，即使当前结构仍远差于初始 reference，只要本步比上一步稍有恢复，仍可
获得很高 step reward。这给“先破坏、再恢复”和来回突变留下了空间。根据相邻序列
变化恢复出的动作统计：

| 行为 | A12 | A36 |
|---|---:|---:|
| 再次修改已访问位置 | 68.27% | 64.29% |
| 连续修改同一位置 | 59.75% | 54.53% |
| 两步后回到原序列 | 51.74% | 44.72% |

A36 的多 actor 去相关使循环略有减少，但比例依然很高。当前
`prevent_revisit_positions=false`，动作空间没有阻止这种 reward exploitation。

### 5. 当前 total reward 不是力学性能改善量

现有 episode 指标是：

```text
total_reward = sum(step_reward_1 ... step_reward_24) + terminal_absolute_zscore
```

terminal reward 不是 `property(final)-property(initial)`。命中预测结构时，代码还使用
按 source PDB stem 找到的静态 predicted structure，它并非 terminal mutated sequence
的新预测结构，其固定分量会进一步减弱动作与 reward 的因果关系。

### 6. 没有能够判断策略效果的固定验证指标

A12/A36 都配置为 `validate_every=0`、`validation_episodes=0`。目前只有训练期间由
epsilon-greedy、stale actor policy、不同初始 PDB 产生的 reward 曲线。日志也没有在
episode summary 中单独保存 initial/final strength、toughness 和二者的 delta。因此，
即使某些 checkpoint 已经学到局部有效策略，现有曲线也无法可靠识别。

### 7. sequence-only observation 与结构/history reward 不匹配

ESM2 观测主要包含当前序列，但 reward 和可行动作还依赖当前 PyRosetta Pose、初始
reference、previous pose、visited positions 和 remaining horizon。同一序列可能因
不同结构历史对应不同 reward/target，当前状态不是充分 Markov 状态。这会提高 TD
目标方差，并限制 Q head 学习稳定的 action ranking。

## 应该如何解决

### P0：先建立固定、配对的策略验证协议

这是下一步最高优先级，不应先继续完整训练或调 DDQN 超参数。选择固定的 128–256 个
validation PDB，对 A12/A36 的多个历史 checkpoint 执行 `epsilon=0` greedy evaluation，
每个 PDB 使用完全相同的初态和 horizon，并保存：

```text
initial/final strength and delta_strength
initial/final toughness and delta_toughness
initial/final terminal score and delta_terminal
collision excess over reference
accepted/repeated/reversed actions
discounted realized return and predicted Q
```

同时评估 random policy、untrained policy 和简单启发式 policy，报告 paired mean、
median、improved fraction、top-k enrichment 及 bootstrap 95% CI。先用该验证器检查现有
checkpoint，才能判断“模型完全没学到”还是“训练 total reward 指标看不出来”。

### P1：将 terminal reward 改为相对初态的 improvement

建议基础形式为：

```text
terminal_reward = 0.5 * [z(strength_final) - z(strength_initial)]
                + 0.5 * [z(toughness_final) - z(toughness_initial)]
```

训练阶段先只使用与动作真正对应的 PyRosetta terminal pose。原始 source sequence 的
静态 ColabFold 结构不能代表 mutated terminal sequence，不应固定占终态评分 50%。
ColabFold 可放到离线验证阶段：只对候选 terminal sequences 重新预测结构，用于二次
确认和不确定性分析。

### P2：重构 step reward 为弱、零中心、难以循环利用的 shaping

step reward 的职责应是约束搜索过程，而不是压过 terminal objective：

1. 将中性变化映射到 0，而不是 0.5；按 horizon 缩放，使 24 步 shaping 总量明显小于
   或至多接近一个有意义的 terminal improvement。
2. collision 同时惩罚 `delta_from_previous` 和 `excess_over_reference`；RMSD 同时约束
   previous 与 reference，避免“先破坏后恢复”得分。
3. 优先采用 potential-based shaping：`r_shape = gamma * Phi(s') - Phi(s)`，其中
   `Phi` 是相对 initial reference 定义的结构质量势函数。这样循环轨迹不会持续积累
   正 reward，也更不容易改变原 terminal objective 的最优策略。
4. 启用位置访问 mask，基础版本中每个 residue 每 episode 最多修改一次；至少禁止
   immediate revisit 和 exact reversal。

### P3：补全 Markov state

在 per-residue ESM2 embedding 之外加入：visited-position mask、remaining-step
fraction、当前位置结构质量摘要，以及 initial-to-current collision/RMSD/H-bond delta。
若完整结构表征代价过高，先加入这些低维量也能显著减少状态混叠。

### P4：在 reward 对齐后再改价值学习

只有 P0–P3 完成后，才值得继续处理 Q 高估和稀疏 terminal credit：

1. 增加 4/8-step return 或按 episode 采样，提高 terminal signal 向前传播速度。
2. 对 terminal transitions 做有上限的分层采样，而不是无限放大少量终止样本。
3. 记录 Q-versus-realized-discounted-return calibration，并按 remaining horizon 分组。
4. 再比较 learning rate、target sync、policy sync、F/G；这些应是第二层问题，而不是
   当前最先解决的问题。

## 下一步应该聚焦什么

下一阶段的唯一主问题应表述为：

> **怎样让 agent 在固定未见蛋白上，稳定提高相对初始结构的 predicted strength 和
> toughness，而不是提高由局部安全分主导的累计 reward？**

建议按以下短周期推进：

1. 用现有 A12/A36 checkpoints 建立固定验证基线，确认是否存在任何隐藏的 terminal
   improvement；这一步不需要重新训练。
2. 实现 delta terminal reward、零中心 potential shaping、禁止重复位置和状态补充。
3. 用小规模固定数据做 5–10k episode 消融：terminal-only、terminal+shaping、是否加入
   visited/horizon state。以验证集 `delta_strength/delta_toughness` 为主指标。
4. 只有某个版本在 paired validation 上显著优于 random/untrained baseline，且置信区间
   不跨 0，才启动更长训练；actor 数量使用 A12 或当前资源下更稳妥的中等配置即可。

最终判断：A36 没有显示出优于 A12 的策略质量，只是以更高资源代价取得有限吞吐和
轻微数值稳定性改善。当前优先级应从“完成 64 epochs”转为“证明 reward 与力学性能
改进一致”。在这个问题解决前，更多 epochs 主要增加计算量，不增加结论可信度。

## 2026-08-13：terminal mechanical improvement reward 与 A24 吞吐分析

### 修改目标

本轮先不重构 step shaping，而是完成两个可独立验证的改动：

1. 将结构力学 terminal reward 从终态绝对预测值改为相对同一 episode 初态的改善量；
2. 保持 strength/toughness 在 terminal objective 内部等权，同时提高 terminal reward
   相对 24 个 step reward 的整体权重。

### Reward 定义

random forest artifact 中保存的训练集 target mean/std 被用于 z-score。原始 terminal
improvement 定义为：

```text
delta_strength_z  = z(strength_final)  - z(strength_initial)
delta_toughness_z = z(toughness_final) - z(toughness_initial)

raw_terminal_reward = 0.5 * delta_strength_z
                    + 0.5 * delta_toughness_z
```

环境最终注入 transition 的 terminal reward 为：

```text
terminal_reward = terminal_reward_scale * raw_terminal_reward
```

本轮将 `--terminal-reward-scale` 默认值由 `1.0` 提升为 `8.0`。因此 strength 与
toughness 的相对比例仍严格为 `1:1`，只是二者合成后的 terminal objective 整体放大
8 倍。该参数仍可由启动命令覆盖，便于后续做 `1/4/8/16` 消融，而不需要再次修改代码。

### 初态和终态的结构来源

- `initial`：`MechanicalProteinEnv.reset()` 后保存的 `reference_pose`；
- `final`：相同 episode 最后一步 repack 后的 `current_pose`；
- 两者使用相同的 hbond/topology feature extractor 和相同 random forest artifact 预测；
- 当前训练 reward 不再混入按原始 source PDB 文件名找到的静态 ColabFold 结构。

排除静态 ColabFold 结构的原因是：它对应原始序列，而不是 episode 完成 24 次 mutation
后的 terminal sequence。如果把它固定占 50%，该分量与本 episode 的 action 无关，会
稀释 credit assignment。对 terminal mutated sequence 重新运行 ColabFold 仍适合放在
离线候选验证阶段，但不适合直接阻塞当前在线 actor。

### 代码修改

1. `model/reward_module/terminal_reward/calculator.py`
   - 新增 `MechanicalImprovementTerminalRewardCalculator`；
   - 新增包含 initial/final prediction、两项 z-score delta 和 reward components 的结果类；
   - 保留旧 dual-structure calculator 作为离线评估/兼容 API，不再作为默认训练 reward。
2. `model/environment_module/environment.py`
   - episode finalize 时同时传入 `reference_pose` 和 `current_pose`；
   - 环境层继续统一应用 `terminal_reward_scale`。
3. `training.py` 与 `model/asynchronous_module/runtime.py`
   - 串行和异步 actor 都改用 improvement calculator；
   - 默认 scale 设为 `8.0`，并增加 finite/non-negative 参数检查；
   - 保留 `--terminal-predicted-pdb-dir` 仅用于旧命令行兼容，默认训练路径不再使用它。
4. `train_version_terminal_async.sh`
   - 新增环境变量 `MPRL_TERMINAL_REWARD_SCALE`，默认 `8.0`；
   - 启动目录名称记录 `terminalx<scale>`，防止不同 reward 实验混淆；
   - 启动命令显式传递 `--terminal-reward-scale`。
5. `model/logging_module/training_logger.py`
   - episode JSONL/CSV 与 TensorBoard 新增 initial/final strength、toughness；
   - 新增 `mechanical_delta_strength_z` 和 `mechanical_delta_toughness_z`；
   - 后续判断训练效果应优先观察两项 delta，而不是只观察 total reward。

### Scale 8 的小样本依据

使用真实 PyRosetta pose 做了两层 smoke test：

1. 单个 PDB、单次 mutation 的完整环境路径可以正常计算 initial/final reward。该样本的
   7 个 RF feature 恰好没有改变，因此 delta 为 0。这不是代码错误，而是当前 7-feature
   predictor 对某些单点侧链变化不敏感的直接表现。
2. 对 8 个真实 PDB 分别执行 24 次随机 mutation/repack，8/8 都得到非零 delta。raw
   terminal reward 范围约为 `[-0.1964, 0.1838]`；scale 8 后约为
   `[-1.5711, 1.4708]`。

当前旧训练中 24 个 step reward 合计通常约 16。scale 8 已经使 terminal signal 从原先
约 `0.1` 量级提升到可检测的 `1` 量级，但还没有一次性压过全部 step shaping，因此可作
为“先调整权重”的保守基线。是否还需要提高，应由固定验证集上的两项 mechanical delta
决定，而不是仅凭训练 total reward 决定。

### 测试结果

```text
targeted tests:
39 passed in 4.27s

full repository tests:
120 passed, 2 skipped in 9.16s

bash -n train_version_terminal_async.sh: passed
git diff --check: passed
```

PyRosetta smoke 中仍观察到 artifact 由 scikit-learn `1.6.1` 训练、当前环境为 `1.7.2`
的 `InconsistentVersionWarning`。本轮预测可运行且输出有限值，但正式长训练前最好在当前
环境重新导出 artifact，或将运行环境固定为 `scikit-learn==1.6.1`，消除序列化兼容风险。

### A24 速度分析

分析日志：

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
async_a24_esm8_f8_g1_20260812_162817/train_stdout.log
```

为排除启动、数据清洗和尾部停止时间的影响，A12/A24/A36 都使用相同的
`global_step=50,000 -> 750,000` 区间比较：

| actors | environment steps/s | episodes/day（24 steps） | 492,864 episodes 预计耗时 |
|---:|---:|---:|---:|
| 12 | 8.1613 | 29,381 | 16.78 days |
| 24 | 9.6780 | 34,841 | 14.15 days |
| 36 | 9.5636 | 34,429 | 14.32 days |

A24 相对 A12 吞吐提高 `18.58%`；A24 相对 A36 还高 `1.20%`，同时少使用 12 个 actor。
所以在当前服务器、3 个 ESM inference worker 和一个 learner 的配置下，A24 是三者中
更合理的资源/吞吐折中点。

actor 自身 episode elapsed 也显示出明显资源竞争：

| actors | mean episode sec | median | p90 |
|---:|---:|---:|---:|
| 12 | 23.22 | 22.36 | 30.60 |
| 24 | 48.72 | 47.12 | 59.73 |
| 36 | 78.41 | 75.47 | 95.83 |

从 A12 到 A24，actor 数量翻倍，但单 actor episode 延迟也约翻倍；A36 延迟继续增加。
说明 PyRosetta CPU/内存带宽、ESM inference queue 或进程调度已经饱和，增加 actor 主要
增加并发等待，而不是产生线性吞吐。A24 当前日志在约 22.4 小时完成 32,641 episodes；
因为尚未达到完整 24 小时，表中的 34,841/day 是稳态区间外推，不把它误写为完整一天
的实测值。

### 重要运行说明与下一步判断

当前正在运行的 `async_a24_esm8_f8_g1_20260812_162817` 在本次代码修改前启动，所以它
仍使用旧的绝对 terminal reward 和 scale 1；运行中的 Python 进程不会自动加载新代码。
该 run 仍可作为 A24 速度基线，但不能用来评价新的 mechanical improvement reward。

下一次新启动的 run 才会使用 delta reward 和 scale 8。建议先做短周期对照，并以
`mechanical_delta_strength_z`、`mechanical_delta_toughness_z`、二者 improved fraction
及固定 validation PDB 的 paired delta 为主要判据。如果 scale 8 仍无法产生可检测改善，
再进入已提出的 step reward 零中心化/potential shaping 重构，而不是继续只提高 actor
数量或盲目延长训练。

## 2026-08-13：DDQN Q-value 高估的原因、诊断与解决顺序

### 首先区分“Q 很大”和“Q 高估”

Q-value 的定义是从当前 transition 开始的期望折扣回报：

```text
Q(s_t, a_t) = E[r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...]
```

因此 Q 数值大本身不能直接证明高估。正确的诊断对象不是完整 episode total reward，
而是同一个 `(s_t, a_t)` 对应的实际 discounted return：

```text
G_t = r_t + gamma*r_(t+1) + ... + gamma^(T-t)*r_T
calibration_error = Q(s_t, a_t) - G_t
```

不过当前成熟 run 中 online Q 约为 `79-81`、target Q 约为 `75-78`，而一个完整 episode
的未折扣 total reward 通常只有 `15-16`。随机抽到的中间 transition 所剩余的真实 return
还会更小，所以当前 Q-value 仍然高度可疑。DDQN 只能缓解标准 DQN 中由同一网络同时
选择和评估最大动作造成的统计高估，不能消除 reward、状态、数据和 bootstrapping 带来
的其他系统性偏差。

### 可能原因与对应方案

| 原因 | 当前项目中的具体风险 | 对应解决方案 |
|---|---|---|
| Step reward 长期为正 | 无明显改善也可能获得约 `0.5` 的中性分，24 步持续积累正 reward | 将 step reward 零中心化；无改善为 0、恶化为负；降低 step reward scale |
| Reward exploitation | 重复修改、先破坏再恢复也可能持续获得相对 previous pose 的局部分数 | 禁止重复位置和立即反向突变；使用相对 initial reference 的 potential-based shaping |
| 状态不含剩余步数 | 同一序列在第 1 步和第 23 步的未来 return 不同，但网络无法区分 | 加入 `remaining_steps / max_steps` |
| 状态缺少结构和历史 | ESM2 主要表示序列，而 reward 还依赖 Pose、reference、visited positions | 加入 visited mask、collision/H-bond/RMSD 相对初态的结构摘要 |
| Bootstrapping 误差传播 | 偏高的 target Q 通过 `r + gamma*Q_target` 反复向前传播 | 使用 4/8-step return；适度降低 gamma；对异常 TD target 做有依据的范围监控/裁剪 |
| Online/target 误差相关 | target 定期完整复制 online，两个网络并不真正独立 | 比较 soft/Polyak update；进一步可测试双 critic、ensemble 或 clipped/min-Q |
| Hard sync 阶梯效应 | 早期同步后 loss/TD error 会显著跳升 | 比较 `N=63/125/250`；或使用 `tau=0.005-0.01` 的 soft update |
| Update-to-data ratio 不合适 | 相同 replay 数据被过度复用，有限行为分布上的误差被放大 | 调整 F/G、降低 UTD；提高数据多样性；按相同 environment steps 对比 |
| 异步策略陈旧 | replay 同时包含不同 actor policy version 的 transition | 缩短 actor policy sync；记录 policy age；过滤或降低过旧 transition 的权重 |
| 未覆盖动作的外推高估 | 巨大突变动作空间中，少采样动作也可能被网络预测为高 Q 并被 argmax 选中 | 更均衡探索；ensemble uncertainty；conservative Q regularization |
| Terminal reward 稀疏 | 力学 improvement 只在最后一步出现，很难影响前面的动作 | n-step return；有上限的 terminal/episode-aware sampling |
| 力学预测器噪声 | RF 对部分突变不敏感，对另一些结构特征变化可能跳变 | 记录预测不确定性；限制异常 delta；对候选结构进行重复或离线验证 |
| 学习率或梯度过大 | Online Q 快速追逐不断移动的 bootstrap target | 测试 `1e-4 -> 5e-5`；保留 Huber loss 和 gradient clipping |
| 表征/网络泛化误差 | 共享 per-residue head 需要覆盖大量异质蛋白和变长动作空间 | 加入 LayerNorm/正则化/ensemble，并补充结构条件输入 |

### 当前终止状态处理检查

Replay Buffer 当前使用：

```text
done = terminated or truncated
```

TD target 使用：

```text
target = reward + gamma * (not done) * next_q_target
```

当前 `max_steps=24` 是任务定义的有限 horizon；最后一步产生 terminal reward，并停止
继续 bootstrap，因此把该 truncated transition 当作 done 是合理的。若以后 truncated
仅表示外部时间限制、任务本身仍可继续，则应区分 terminated 和 time-limit truncation；
但当前实现暂时不像 Q 高估的主要来源。

### Target network 同步频率的角色

当前异步设置 `F=8, G=1, target_sync_interval=125`，即每约 1000 个 environment
transitions 硬同步一次。日志表明：

- 前 2k optimizer updates，同步后 TD error 可增加约 `90%-120%`；
- 2k-5k updates 降为约 `30%-40%`；
- 5k-10k updates 降为约 `9%-17%`；
- 10k updates 以后通常只有约 `0%-2%`。

所以 hard sync 是早期阶梯增长的影响因素，但没有证据表明 `N=125` 在成熟阶段持续
制造发散。历史 `F8/G1, N=63` run 的单次同步跳变反而比 `N=125` 小，说明硬同步间隔
越长，online-target 参数差可能积累得越大，然后一次性释放。不能简单认为“同步越频繁
越不稳定”或“同步越少越稳定”。

更重要的是，online Q 与 target Q 可以一起收敛到错误的高估固定点。两条 Q 曲线彼此
接近、loss 下降，只能说明 Bellman 自洽性改善，不能证明 Q 接近真实 return，也不能
证明策略提高了 strength/toughness。

### 建议的解决优先级

#### P0：建立 Q-versus-return calibration

对完整 episode 反向计算每个 transition 的 `G_t`，保存：

```text
predicted_q
realized_discounted_return
q_minus_return
remaining_steps
terminal/non-terminal
actor_policy_age
```

按 remaining horizon、PDB、训练阶段和 terminal proximity 分组，报告 calibration bias、
MAE、RMSE 和散点图。这是确认“高估多少、从哪里开始高估”的必要步骤。

#### P1：补全 Markov state

至少加入：

```text
remaining_step_fraction
visited_position_mask
collision_excess_over_initial
hbond_delta_from_initial
rmsd_from_initial
```

否则相同 ESM2 sequence embedding 可能对应不同 remaining horizon、结构历史和可行动作，
网络被迫用一个 Q 值拟合多个真实 return。

#### P2：使 step shaping 零中心并限制循环利用

将无变化对应的 reward 设为 0，恶化为负，改善为正；step shaping 的 episode 总幅度应
明显小于或至多接近有意义的 terminal mechanical improvement。优先采用：

```text
r_shape = gamma * Phi(s_next) - Phi(s)
```

其中 `Phi` 相对 initial structure 定义。这样循环回到原状态不会持续产生正收益。

#### P3：使用 n-step return

建议先测试 `n=4` 和 `n=8`：

```text
target_n = r_t + gamma*r_(t+1) + ...
         + gamma^(n-1)*r_(t+n-1)
         + gamma^n*Q_target(s_(t+n))
```

它可以减少反复 bootstrap 的次数，并让 terminal mechanical reward 更快传播到前面的
mutation action。

#### P4：再比较 target update 和优化超参数

在前述问题处理或至少具备 calibration 指标以后，再做相同数据、相同 environment steps
的消融：

```text
hard sync N = 63 / 125 / 250
soft update tau = 0.005 / 0.01
learning rate = 1e-4 / 5e-5
```

主要判据应是 Q-return calibration 和固定验证集的 `delta_strength/delta_toughness`，
loss/TD error 仅作为数值稳定性指标。

### 最终判断

当前 Q 高估更可能由“持续正 step reward + 非充分状态 + bootstrapping + reward exploitation”
共同导致，而不是由单一 target-sync interval 导致。推荐执行顺序为：

```text
Q-return calibration
-> horizon/history/structure state
-> step reward zero-centering and potential shaping
-> n-step return
-> soft target update
-> learning rate and F/G tuning
```

最终目标不是让 online Q 和 target Q 相互接近，而是让两者同时接近真实 discounted return，
并让 greedy policy 在固定未见蛋白上产生可重复的正 `delta_strength` 和
`delta_toughness`。

## 2026-08-14：terminal×8 结果分析与 PER、零中心 shaping、3-step return

### 分析对象

新 terminal reward run：

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
async_a24_esm8_f8_g1_terminalx8.0_20260813_150454/
```

对照 run：

```text
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/
async_a24_esm8_f8_g1_20260812_162817/
```

两个 run 都使用 A24、ESM batch 8、F8/G1、batch size 128、horizon 24 和
`target_sync_interval=125`。主要差异是旧 run 使用绝对 terminal reward、scale 1；新 run
使用 final-minus-initial mechanical delta、scale 8。因此可在相同 optimizer update 区间
比较数值收敛形态，但异步采样仍不是完全确定性的配对实验。

### 模型收敛分析

#### 1. 修改 terminal reward 没有改变基本的“先升后降”形态

| optimizer 区间 | run | loss | mean abs TD | grad norm | mean Q | mean target Q |
|---|---|---:|---:|---:|---:|---:|
| 0-2k | old | 0.382 | 0.634 | 1.921 | 7.72 | 7.53 |
| 0-2k | delta×8 | 0.408 | 0.664 | 1.973 | 7.73 | 7.53 |
| 5k-10k | old | 6.534 | 6.983 | 19.749 | 109.35 | 105.59 |
| 5k-10k | delta×8 | 8.107 | 8.566 | 21.626 | 127.06 | 122.71 |
| 10k-20k | old | 6.469 | 6.855 | 20.807 | 136.60 | 131.18 |
| 10k-20k | delta×8 | 7.160 | 7.558 | 22.868 | 148.67 | 142.75 |
| recent 10k | old | 3.370 | 3.639 | 10.232 | 79.41 | 76.22 |
| recent 10k | delta×8 | 3.266 | 3.537 | 10.319 | 76.40 | 73.32 |

两者都经历 Q/loss/TD/gradient 先快速升高、再逐渐下降。delta×8 在 5k-10k 阶段的
loss 和 TD error 分别比旧 run 高约 24% 和 23%，Q 峰值也更高。这说明放大一次性的
terminal delta 增加了早期 target variance，却没有改变由正 step reward 和 bootstrapping
主导的整体动力学。进入较晚阶段后，两者几乎回到相同数值范围；数值上能够收敛，但这
只说明 learner 再次达到 Bellman 自洽，并不说明策略学会力学性能改善。

#### 2. Target hard sync 不是两条曲线相似的唯一原因

新旧 run 都使用相同 `N=125`，早期同步后均有明显 TD 阶梯，10k updates 后同步边界
影响降至约 0%-2%。terminal×8 没有消除该现象，也没有造成长期新增发散。因此当前
“训练形态没有区别”主要说明 terminal transition 在 uniform one-step replay 中占比和
传播能力仍不足，而不是 terminal reward 代码没有生效。

### 训练效果分析

#### 1. Episode total reward 的早期增长依旧来自正 step reward

```text
old run recent mean step reward:       0.6668
delta×8 recent mean step reward:       0.6592
delta×8 recent terminal/transition:   -0.0094
```

24 个约 0.66 的 step reward 可以贡献约 15.8，而 terminal reward 每 episode 只出现一次。
即使 terminal scale 提高到 8，它仍被旧的正向 shaping 基线和 one-step uniform sampling
稀释。TensorBoard 中 episode total reward 先升后平台，不能视为力学性能增长。

#### 2. Strength 没有学习趋势，toughness 仍为负改善

delta×8 run 按每 2000 episodes 分块：

```text
episodes 0-2000:
  delta_strength_z  = -0.0037, positive = 51.0%
  delta_toughness_z = -0.1046, positive = 26.3%
  terminal_reward   = -0.4331

episodes 2000-4000:
  delta_strength_z  = -0.0050, positive = 48.8%
  delta_toughness_z = -0.0577, positive = 33.4%
  terminal_reward   = -0.2507

episodes 24000-26000:
  delta_strength_z  = -0.0045, positive = 48.3%
  delta_toughness_z = -0.0506, positive = 34.8%
  terminal_reward   = -0.2203
```

toughness 的恶化程度在探索早期有所减小，但约 4000 episodes 后主要在负值附近平台；
strength 始终接近随机正负各半，没有稳定正趋势。此时 epsilon 已接近下限，后续平台更能
代表当前 greedy/near-greedy policy 的效果。结论是：terminal reward 修改已生效，但
uniform one-step learner 没有把该信号可靠传播到前面的 mutation action。

### 实现 1：Ape-X 风格 TD-error prioritized replay

#### 采样与更新公式

新 replay 模式使用 learner 计算的 n-step TD error：

```text
priority_i = abs(td_error_i) + priority_epsilon
P(i) = priority_i^alpha / sum_j(priority_j^alpha)
w_i = (N * P(i))^(-beta)
w_i <- w_i / max_batch(w)
```

逐样本 Huber loss 乘以 `w_i` 后再求 batch mean。每次 optimizer update 完成后，learner
将最新绝对 TD error 回写到对应 replay index。新 transition 使用当前最大 priority，
保证至少能尽快被 learner 看见一次。beta 从 start 线性退火到 1，以逐步修正 prioritized
sampling 引入的分布偏差。

新增参数：

```text
--replay-sampling {uniform,prioritized}  # CLI 默认 uniform
--priority-alpha 0.6
--priority-beta-start 0.4
--priority-beta-end 1.0
--priority-beta-steps 1000000
--priority-epsilon 1e-6
```

`uniform` 模式使用全 1 importance weights，与旧 loss 路径数值等价。Replay snapshot
format 升级到 v2，保存 priority、n_steps、beta 配置和 RNG；v1 snapshot 可作为 uniform、
one-step replay 兼容加载。

本实现准确地采用 Ape-X 的 TD-error priority、central replay 和 importance weighting，
但初始 priority 由“当前最大 priority”提供，第一次 learner update 后才变为真实 TD error。
原因是当前 CPU actor 只持有 online policy，没有 target network；让 actor 额外计算 DDQN
TD error 会增加模型同步和 CPU 开销。对于当前集中式 learner，这是更稳妥的 Ape-X-style
实现，而不是宣称复现论文的全部分布式系统细节。

### 实现 2：零中心、弱 step shaping

原诊断 unit score 继续保留在 0-1，便于与历史 TensorBoard 对比；用于训练的标量改为：

```text
centered_collision = collision_unit_score - 1
centered_rmsd      = rmsd_unit_score - 1
centered_hbond     = 2 * hbond_unit_score - 1

step_reward_raw = weighted_mean(centered components)
```

因此：

- collision/RMSD 无惩罚为 0，恶化为负；
- H-bond 无变化为 0，增加为正、减少为负；
- “没有变坏”不再持续产生正 reward。

`--step-reward-scale` 默认及异步脚本设置为 `0.025`。centered weighted mean 的绝对值
不超过 1，所以 24 个合法 action 的 shaping 理论绝对上限约为：

```text
24 * 0.025 = 0.6
```

这低于 pilot 中 scale 8 后约 `1` 量级的有意义 terminal mechanical improvement。非法
action/update error 的显式 penalty 不属于 shaping bound；action mask 应使非法 action
极少进入正常策略路径。

### 实现 3：可配置 n-step return

新增参数：

```text
--n-step N  # CLI 默认 1，异步训练脚本设置 3
```

中央收集端为每个 actor 独立维护 episode-local queue：

```text
R_t^(n) = r_t + gamma*r_(t+1) + ... + gamma^(n-1)*r_(t+n-1)
target  = R_t^(n) + gamma^n * (not done) * Q_target(s_(t+n), a*)
```

episode 终止时 flush 剩余前缀，因此 `n=3` 的最后三个 replay row 分别具有实际
`n_steps=3/2/1`。terminal mechanical reward 会进入最后三个 mutation action 的 return，
不再只监督最后一个 action。Optimizer 调度仍按真实 environment transitions 计数，不按
一次 terminal flush 产生的 replay row 数计数。

PER 的 priority 使用上述 n-step target 对应的 TD error，所以两项功能在数学上保持一致。

### 修改文件

```text
model/replay_buffer_module/replay_buffer.py
model/replay_buffer_module/n_step.py
model/replay_buffer_module/__init__.py
model/agent_module/ddqn_agent.py
model/reward_module/reward_calculators.py
training.py
asynchronous_training.py
train_version_terminal_async.sh
tests/test_n_step.py
tests/test_replay_buffer.py
tests/test_ddqn_agent.py
tests/test_reward_calculators.py
tests/test_training_multi_gpu.py
```

Optimization JSONL/TensorBoard 还会自动记录：

```text
mean_importance_weight
mean_sampling_probability
priority_beta
mean_n_steps
```

### 启动配置

`train_version_terminal_async.sh` 当前默认实验配置为：

```text
actors=24
F=8, G=1
replay_sampling=prioritized
priority_alpha=0.6
priority_beta=0.4 -> 1.0
n_step=3
step_reward_scale=0.025
terminal_reward_scale=8.0
```

可通过以下环境变量做消融：

```bash
MPRL_REPLAY_SAMPLING=uniform
MPRL_N_STEP=1
MPRL_STEP_REWARD_SCALE=0.025
MPRL_TERMINAL_REWARD_SCALE=8.0
```

新的 reward/replay 语义与旧 replay 数据不一致，正式实验应从新 replay 开始，不应把旧的
正基线 one-step replay 恢复到新 run 中。

### 测试结果

```text
PER/n-step/reward/agent/asynchronous targeted tests:
85 passed, 2 skipped

PyRosetta environment/reward smoke:
6 passed

full repository:
130 passed, 2 skipped in 9.60s

bash -n train_version_terminal_async.sh:
passed

git diff --check:
passed
```

组合 learner smoke 使用两个变长短 episode，经过 n=3 聚合和 prioritized replay 执行真实
DDQN update：

```text
replay_size=8
loss=0.166277
mean_abs_td=0.561478
mean_n_steps=2.25
priority_beta=0.4000048
priorities_changed=6
finite=true
```

`mean_n_steps=2.25` 正确反映每个四步 episode 的 `3/3/2/1` 聚合；priority changed 少于
8 是因为 prioritized sampling with replacement 允许同一 index 在 batch 中重复。

### 下一轮训练的判断标准

PER 会优先学习当前 TD error 大的 transition，但“大 TD error”不必然等于“有益的力学
样本”；它也可能来自 noisy/outlier reward。因此下一轮不能只期待 loss 更快下降，应同时
检查：

1. `mechanical_delta_strength_z` 和 `mechanical_delta_toughness_z` 的 rolling mean；
2. 两项分别和同时为正的 improved fraction；
3. terminal reward 在 priority 分布和 sampled batch 中的占比；
4. Q-versus-realized n-step/episode return calibration；
5. priority p50/p90/p99，防止少量异常结构长期垄断 replay；
6. 与 `uniform+n1`、`uniform+n3`、`prioritized+n1` 的短周期消融。

本轮修改直接处理了三个已确认的问题：重要 transition 利用不足、step reward 正基线、
terminal credit 传播过慢。它们比继续单独增大 terminal scale 更有针对性；但是否真正提高
未见蛋白的 strength/toughness，仍必须由固定 validation PDB 的配对 mechanical delta
验证，而不能由训练 episode total reward 或 loss 单独判断。

## 2026-08-17：PER + 3-step + zero-centered shaping 训练结果分析

### 分析对象与对齐方式

对比日志：

- 上一版本：`async_a24_esm8_f8_g1_terminalx8.0_20260813_150454`；
- 当前版本：`async_a24_esm8_f8_g1_prioritized_n3_stepx0.025_terminalx8.0_20260814_103321`。

上一版本在 `659,512` environment transitions、`27,464` episodes 后停止；当前版本在本次
分析快照时已到约 `1,682,000` transitions、`70,081` episodes，即约 `9.1` 个完整 epoch。
优化器指标首先在两者共同拥有的 `4,096--659,000` transition 区间比较，共约 `81.9k`
次 optimizer updates；策略效果同时按 epoch 聚合，并按 `source_pdb` 做跨 epoch 配对，减少
不同蛋白难度差异造成的混淆。

### 数值收敛对比

共同训练区间的统计如下。括号中的 p99 比单个偶发最大值更适合表示典型峰值。

| 指标 | 上一版本 mean / p99 / max | 当前版本 mean / p99 / max |
| --- | ---: | ---: |
| loss | 4.485 / 12.129 / 19.906 | 0.175 / 1.722 / 2.655 |
| mean absolute TD error | 4.807 / 12.558 / 20.328 | 1.047 / 6.752 / 8.675 |
| grad norm | 14.233 / 38.130 / 67.954 | 0.346 / 2.128 / 4.006 |
| mean Q | 96.688 / 159.674 / 165.418 | 1.981 / 14.743 / 16.505 |
| mean target Q | 92.869 / 154.448 / 162.868 | 1.351 / 10.378 / 12.805 |

因此，新版本确实显著降低了 Q、loss、TD error 和 gradient 的峰值与方差，也没有再出现
先前接近数值发散的轨迹。PER、3-step return 和 reward 重标定后的 Bellman 回归稳定性明显
更好。

但不能把全部降幅都解释为“策略学得更好”：上一版本每步 reward 存在约 `+0.66` 的正
基线，而当前 step shaping 被零中心化并乘以 `0.025`。reward 单位改变会直接缩小 Q、target
和 loss。此外，PER loss 乘了 importance-sampling weight；当前后期 beta 已达到 `1.0`，
mean importance weight 约 `0.126`，所以加权 loss 很小并不等于未加权 TD error 已消失。
在 `1.30M--1.68M` transitions，loss mean 约 `0.0050`，但未加权 mean absolute TD error
仍约 `0.380`。

当前后期 mean Q 约 `0.321`、mean target Q 约 `0.147`，而实际 episode total reward 和
terminal reward 的均值仍为负。由于 Q 是最优动作的期望回报，不能直接用两者均值证明
高估，但这至少说明仍需做 `Q(s,a)` 与相同 transition 的 realized discounted return 校准；
不能仅凭 Q 的绝对值从 150 降到 10 以下就断言高估已经完全解决。

### 力学优化效果

当前版本按完整 epoch 聚合的结果如下：

| epoch | total reward | terminal reward | delta strength z | delta toughness z | epsilon mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.488 | -0.396 | -0.0061 | -0.0930 | 0.177 |
| 3 | -0.429 | -0.341 | -0.0052 | -0.0801 | 0.050 |
| 5 | -0.379 | -0.292 | -0.0040 | -0.0691 | 0.050 |
| 7 | -0.326 | -0.241 | -0.0039 | -0.0563 | 0.050 |
| 9 | -0.302 | -0.219 | -0.0050 | -0.0498 | 0.050 |

曲线肉眼看起来近似水平，但按相同 PDB 配对后可以检测到弱改善。第 9 epoch 相对第 1
epoch：

- terminal reward 平均增加 `+0.1776`，中位数增加 `+0.1362`；`60.0%` 的 PDB 有改善；
- delta toughness z 平均增加 `+0.0432`，`61.5%` 的 PDB 有改善；
- delta strength z 只增加 `+0.0012`，改善比例 `51.0%`，接近随机波动；
- terminal reward 为正的 episode 比例由第 1 epoch 的 `31.5%` 增至第 9 epoch 的
  `40.5%`，但多数 episode 仍使预测力学性能下降。

所以当前版本不是“完全没有学习”，而是主要学会了**减轻 toughness 恶化**，还没有学会
稳定地产生正向 toughness improvement，更没有学会提高 strength。total reward 与 terminal
reward 的 epoch 内相关系数约 `0.999`，平均累计 shaping 只有约 `-0.08~-0.09`；这说明
step reward 已不再掩盖 terminal objective，total reward 不增长的核心问题已经转移到策略、
状态表示、探索和 terminal signal 本身。

在相同的前三个 epoch 内，上一版本 terminal reward 从 `-0.282` 变为 `-0.238`，当前版本
从 `-0.396` 变为 `-0.341`。按相同 PDB 配对，第 3 epoch 相对第 1 epoch 的 terminal 提升
分别为 `+0.044` 和 `+0.055`，幅度相近。也就是说，新方案明显提高了数值稳定性，但目前
没有证据表明它在相同样本预算内显著提高了力学优化效率；当前版本较长训练后的改善主要
来自持续训练，而不是前三个 epoch 就表现出更强的策略。

### 当前主要问题

1. **缺少固定 greedy validation。** 训练 episode 同时改变策略、epsilon、PDB 顺序和
   replay 分布。训练曲线只能说明数据收集时的表现，不能可靠评价 checkpoint 的策略提升。
   应在固定 validation PDB 上用 `epsilon=0`、相同 seed 做配对评估，记录 strength/toughness
   mean、median、positive fraction、top-k 和 bootstrap confidence interval。
2. **strength reward 几乎没有可学习变化。** strength delta 长期约为 `-0.004~-0.006`，
   正向概率约 50%。可能是 24 次局部突变和 repack 很少改变 RF 所依赖的全局 topology
   features，也可能是 RF 的分段常数输出使局部变化得到近似零信号。应先统计 terminal
   strength delta 的零值比例、唯一值数量、分位数和单步/多步敏感性，再判断 RL 是否有
   足够的 strength 学习信号。
3. **PER 优先“大 TD error”，不等于优先“正向力学样本”。** 负向 terminal outlier 同样
   会被高频采样，因此当前行为更像学习避免严重 toughness 下降。应记录 positive/negative
   terminal transitions 在 replay、priority top 1% 和 sampled batch 中的占比；必要时采用
   terminal-aware stratified replay，同时保留 TD-error priority 和 IS correction。
4. **n=3 对 24-step horizon 仍然较短。** terminal reward 只直接进入最后三个动作的 target，
   更早动作仍依赖多轮 bootstrapping。可以做 `n=3/6/12` 消融，或为 terminal episode 使用
   backward episodic return；不能只通过继续增大 terminal scale 解决 credit assignment。
5. **观测与结构 reward 不完全 Markov。** Q network 看到的是当前序列的 ESM2 embedding，
   但 terminal predictor 使用当前 PyRosetta 结构的氢键/topology。相同序列可能因突变顺序和
   局部 repack 得到不同结构，且观测没有剩余步数、已访问位置、初始机械属性或当前结构
   特征。建议加入 normalized remaining steps、visited-position mask、initial/current mechanical
   proxies；中长期可让 Q head 融合便宜的结构图或氢键摘要。
6. **动作循环仍然允许。** 当前 `prevent_revisit_positions=false`，虽然当前氨基酸的 no-op
   被 mask，策略仍可执行 `A -> B -> A`。零中心 shaping 降低了循环收益，却没有禁止循环。
   下一次消融应开启 `--prevent-revisit-positions`，并确保 visited mask 进入 observation。
7. **replay 覆盖不足一个 epoch。** capacity `50,000` 只对应约 `2,083` 个完整 episode，
   约为 7,701 个训练蛋白的 27%。对高度异质的变长蛋白，PER 还会进一步压缩有效覆盖。
   需要监控 priority entropy/effective sample size；若内存不允许直接扩大，可按蛋白或 reward
   符号分层保留 terminal transitions。
8. **reward artifact 存在版本告警。** 日志显示 RF 由 scikit-learn `1.6.1` 保存，却在
   `1.7.2` 中加载。它未必造成当前趋势，但会削弱 reward 的可复现性，应使用与训练 artifact
   相同的 sklearn 版本，或在当前环境重新训练并重新验证模型。

### Epsilon 衰减分析

当前实现按**全局 environment transition**线性计算：

```text
progress = min(global_environment_steps / epsilon_decay_steps, 1)
epsilon = 1.0 + progress * (0.05 - 1.0)
```

配置是 `epsilon_decay_steps=50,000`。因此 epsilon 在约 `50,000 / 24 = 2,083` episodes
后达到 `0.05`，仅为 `2,083 / 7,701 = 0.27 epoch`。TensorBoard 若按 episode 或记录序号
显示，会造成“5k 多 steps 才触底”的视觉差异，但从代码和日志看，第二个 epoch 开始前
epsilon 已经固定为 `0.05`。24 个 actor 不应再乘入公式，因为 central global step 已经是
所有 actor transition 的总和；`train_batch_size` 也不应乘入，因为每个 PDB 本身就是一个
episode。

如果目标是用前 4 个 epoch 做线性大范围探索，用户提出的数量级是合理的，但公式中的 `4`
应解释为 4 个 epoch：

```text
epsilon_decay_steps = 7701 proteins * 4 epochs * 24 steps = 739,296
```

采用当前单段线性公式时，epsilon 在第 1/2/3/4 epoch 末约为
`0.7625 / 0.5250 / 0.2875 / 0.0500`。因此下一轮最小改动可设置：

```bash
--epsilon-decay-steps 739296
```

这个修改明显比当前 50k 合理，但“衰减 4 个 epoch”和“前几个 epoch 始终保持高探索”并不
完全相同。更推荐后续增加分段 schedule：先用约 1 epoch 保持较高 epsilon，再在 3--7 个
epoch 内下降到 `0.1`，最后缓慢降到 `0.05`。对于异步 Ape-X 风格采集，还可以让不同 actor
长期使用不同 epsilon：一部分 actor 保持探索，一部分 actor 负责利用，避免所有 24 个 actor
在 0.27 epoch 后同时退化为近乎 greedy。

epsilon 也不能无限拉长。当前随机突变大多产生负 terminal reward，过高 epsilon 会持续向
PER 注入高优先级失败样本。因此应比较 `50k`、`739,296` 和分层 actor epsilon，并统一用
固定 greedy validation 判断，而不是选择训练 total reward 最好看的 schedule。

### 建议的下一步顺序

1. 先实现固定 validation PDB 的 checkpoint greedy evaluation，并增加 Q-versus-realized
   return、positive terminal replay ratio 和 priority effective sample size；这是判断其他
   改动是否有效的前提。
2. 保持当前 PER+n3+reward scale 不变，只把 epsilon decay 改为 `739,296` 做单变量对照；
   更理想的是同时测试 actor-specific epsilon。
3. 开启禁止重复位置，并把 remaining horizon、visited mask 和机械/结构摘要加入状态，修复
   部分可观测和动作循环问题。
4. 在确认 terminal predictor 对局部 mutation 确实有分辨率后，再比较 n=6/n=12 或
   terminal-aware replay。若 predictor 对 strength 几乎不响应，继续调 DDQN 超参数不会解决
   strength 不增长，应先改善 reward model 或动作/结构更新尺度。

## 2026-08-19：延长 epsilon 衰减实验的结果与停止条件

### 分析对象与单变量对照

本次比较两个 A24、ESM batch 8、F8/G1、prioritized replay、n=3、
`step_reward_scale=0.025`、`terminal_reward_scale=8.0` 的运行：

```text
快速衰减对照：
async_a24_esm8_f8_g1_prioritized_n3_stepx0.025_terminalx8.0_20260814_103321
epsilon_decay_steps = 50,000

慢速衰减实验：
async_a24_esm8_f8_g1_prioritized_n3_stepx0.025_terminalx8.0_20260817_142250
epsilon_decay_steps = 739,296
```

其余 run configuration 一致，因此这是目前较干净的 epsilon 单变量对照。分析快照中，慢速
实验约包含 `45.4k episodes / 1.09M transitions / 135.8k optimizer updates`，完成约
5.9 个 epoch。epsilon 在第 4 epoch 结束时降到 `0.05`，之后已有接近两个 epoch 的低
epsilon 数据，可初步判断延长探索是否留下后续收益。日志仍在写入，以下数字以本次快照为准。

### 数值稳定性：出现严重暂态价值发散

慢速 epsilon 实验不只是峰值略高，而是在高探索阶段出现了明显的暂态价值发散：

| 指标 | 快速衰减 max（共同 1.09M transitions） | 慢速衰减 max |
| --- | ---: | ---: |
| loss | 2.655 | 13,542.899 |
| mean absolute TD error | 8.675 | 38,947.988 |
| grad norm | 4.006 | 1,295.766 |
| mean Q | 16.505 | 75,023.969 |
| mean target Q | 12.805 | 60,538.406 |

最严重区间是 `100k--250k transitions`，此时慢速实验 mean Q 为 `43,194`、mean TD
error 为 `18,091`、mean grad norm 为 `432.6`。对应 epsilon 仍约处于 `0.87--0.68`，
24 个 actor 大部分动作均为随机动作。随着 epsilon 继续降低，指标从峰值恢复；但“恢复”
不等于已经收敛到上一版本的稳定范围。

在 epsilon 已触底后的共同 `739,296--1,090,000 transitions` 区间：

| 指标 | 快速衰减 mean | 慢速衰减 mean |
| --- | ---: | ---: |
| loss | 0.0064 | 0.1348 |
| mean absolute TD error | 0.3840 | 2.0436 |
| grad norm | 0.0548 | 9.8064 |
| mean Q | 0.3522 | 3.9640 |
| mean target Q | 0.1790 | 2.9324 |

慢速实验在该阶段仍有 `39.6%` 的 update 的 unclipped grad norm 超过阈值 10；快速对照
为 0。最近 10k updates 中慢速实验已进一步恢复到 `loss=0.081、TD=1.60、grad=6.13、
Q=3.06`，但仍明显高于快速对照成熟阶段的 `0.004、0.35、0.03、0.22`。因此当前状态
应描述为“从严重发散中恢复并继续下降”，不能描述为“已经稳定收敛”。

### 为什么慢探索反而放大 Q

epsilon-greedy 的更多随机动作只增加行为数据，不保证 DQN 更稳定或更接近最优策略。当前
项目同时具备 function approximation、bootstrapping 和 off-policy learning，即经典的
deadly triad；慢衰减进一步触发了以下正反馈：

1. 每条 transition 只监督一个动作，而每个变长蛋白有 `L*20` 个动作。高 epsilon 产生
   大量异质、低回报 transition，Q target 的 `max` 仍会选择缺乏真实监督的高估动作。
2. PER 按绝对 TD error 采样。随机结构中的极端负 reward 或外推 Q 会产生大 TD error，
   随后被反复采样；`priority_alpha=0.6` 和早期 beta 小于 1 会加强这个反馈。
3. 24 个 actor 使用同一 epsilon schedule，在前四个 epoch 同时偏向随机探索，没有一组
   稳定的低 epsilon actor 持续提供较高质量的 exploitation transition。
4. replay capacity 只有 50,000 transitions，约等于 2,083 个 episode、0.27 epoch。
   即使探索偶然发现正向轨迹，也可能在一个完整数据集 pass 之前被覆盖；PER 又可能让少量
   不可约噪声长期占据有效采样质量。
5. observation 不含 remaining horizon、visited positions 和完整结构历史，同一 ESM2
   sequence state 对应多个真实 return，进一步增加 bootstrap target 方差。

梯度裁剪和 Huber loss 避免了 NaN/Inf，日志中也没有 FloatingPointError，但它们只能限制
单次参数更新，不能消除错误 target 和 PER 采样分布导致的价值发散。

### Total/terminal reward 的真实结果

慢速衰减实验按 epoch 聚合如下：

| epoch | mean epsilon | total reward | terminal reward | delta strength z | delta toughness z |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.881 | -0.617 | -0.514 | -0.0059 | -0.1225 |
| 2 | 0.643 | -0.595 | -0.495 | -0.0069 | -0.1170 |
| 3 | 0.406 | -0.536 | -0.441 | -0.0061 | -0.1043 |
| 4 | 0.168 | -0.457 | -0.369 | -0.0054 | -0.0868 |
| 5 | 0.050 | -0.422 | -0.337 | -0.0053 | -0.0790 |
| 6（部分） | 0.050 | -0.409 | -0.323 | -0.0054 | -0.0755 |

表面上 total reward 从 `-0.617` 提高到 `-0.409`，但该变化与 epsilon 从 0.88 降到
0.05 几乎完全重合。训练 episode reward 是当前 epsilon-greedy behavior policy 的表现，
不是固定 greedy policy 的性能；高随机比例本来就应得到更低训练 reward，所以这条曲线
不能证明 learner 随训练变好。

进入 `epsilon<=0.051` 后的约 14.7k episodes：

```text
total reward/order correlation    = 0.0066
terminal reward/order correlation = 0.0062
delta strength/order correlation  = -0.0043
delta toughness/order correlation = 0.0105
```

均接近 0。相同 PDB 在第 5 到部分第 6 epoch 的配对变化为：

```text
total reward delta:       +0.0155 +/- 0.0164 (approx. 95% CI)
terminal reward delta:    +0.0156 +/- 0.0162
delta strength change:    +0.00006 +/- 0.00156
delta toughness change:   +0.00383 +/- 0.00330
```

total、terminal 和 strength 的区间均包含 0，尚无明确改善。toughness 只有边界性的“少
恶化”信号，仍不是稳定正 improvement。

step shaping 已不再掩盖 terminal：各 epoch 中 total 与 terminal 的相关系数约
`0.9993--0.9996`，累计 shaping 只有 `-0.085~-0.103`。因此现在 total reward 不增长
不是 step reward 权重过大，而是 terminal mechanical objective 本身没有被策略有效优化。

strength 尤其停滞：均值一直约为 `-0.005~-0.007`，正向比例约 49%--51%。第 5 epoch
同时改善 strength 和 toughness 的比例仅 `17.85%`。这表明延长随机探索没有创造可检测的
strength 学习信号。

### 与快速衰减对照的直接比较

在相同 PDB、相同 epoch 的配对比较中，慢速衰减在前六个 epoch 的 total 和 terminal
始终更差。第 5 epoch：

```text
                         快速衰减       慢速衰减       slow-fast
total reward              -0.379         -0.422         -0.043
terminal reward           -0.292         -0.337         -0.044
delta strength z          -0.0040        -0.0053        -0.0012
delta toughness z         -0.0691        -0.0790        -0.0099
```

慢速策略在配对 PDB 上取得更高 terminal reward 的比例只有 `47.5%`。到部分第 6 epoch，
terminal gap 仍约为 `-0.065`。虽然异步训练不是逐 action 的确定性复现，但样本量约 7,700
且差异持续多个 epoch，当前没有证据支持“把所有 actor 的线性探索延长到四个 epoch 能提高
力学性能”。

### 当前运行是否还有继续训练的必要

**不建议按当前配置继续完整 64 epoch。** 这次实验已经回答了原假设：统一、长时间的高
epsilon 没有改善同预算下的 terminal/total reward，反而造成严重暂态 Q 发散，epsilon 触底
后近两个 epoch 也未出现明确 reward 趋势。继续训练可能让数值进一步恢复，甚至缓慢接近
快速衰减基线，但这不是该实验原本希望证明的“更充分探索产生更优策略”。

考虑到异步运行不能精确恢复 in-flight tasks/replay，操作上可让当前任务完成第 6 个完整
epoch并保存整齐 checkpoint，然后停止；也可以直接保留最近 checkpoint。不要仅为观察训练
reward 曲线继续跑到 64 epoch。后续是否值得从该 checkpoint 开始新实验，应由固定 validation
PDB 的 `epsilon=0` greedy evaluation 决定：

- 若 epoch 1/4/6 checkpoint 的 paired terminal delta 单调改善且优于快速衰减 checkpoint，
  才有继续训练的依据；
- 若 greedy validation 也无改善，则应把该 run 作为负结果结束，转向 reward/state/action
  设计，而不是继续增加训练时长。

### 当前阶段的主要问题与改进方向

#### P0：先修正评价方式

训练 total reward 混入 epsilon 和不同 actor policy version，不能作为主要模型选择指标。
下一步应固定 128--256 个 validation PDB，对 untrained、快速衰减 epoch 1/4/6 和慢速衰减
epoch 1/4/6 checkpoints 做相同 seed、相同 horizon、`epsilon=0` 的 paired evaluation，记录：

```text
delta_strength_z / delta_toughness_z / delta_terminal
both-positive fraction / top-k enrichment
Q(s,a) - realized discounted return
重复位置与反向突变比例
```

没有这一评估，继续调 epsilon 只是在改变训练数据收集曲线，无法判断真正的 greedy policy。

#### P1：将“统一慢衰减”改为 actor-specific exploration

异步系统的优势是可以让 actor 承担不同角色，而不是 24 个 actor 同时从 epsilon 1.0 线性
降到 0.05。下一组实验应始终保留低 epsilon exploitation actors，同时让一部分 actor 使用
中高 epsilon 收集探索数据。这样 replay 中同时存在策略当前最优附近的数据和新区域数据，
也避免四个 epoch 几乎完全由随机 behavior 主导。

在没有实现 actor-specific epsilon 前，可先比较 1--2 epoch 的中等 decay（约
`184,824--369,648 transitions`），而不是继续使用 739,296。探索策略还可由“动作均匀随机”
改为受约束的 position/substitution exploration，例如禁止重复位置、限制立即反向突变、按
结构可接受性筛选动作；在 `L*20` 巨大动作空间中，这通常比盲目延长 epsilon 更有效。

#### P2：减弱 PER 对异常 TD 的正反馈

当前 PER 优先大绝对 TD，而不是优先正向 terminal improvement。建议做以下消融，而不是
一次全部修改：

1. uniform/PER 混合采样，保证基础覆盖；
2. `priority_alpha` 从 0.6 降至 0.3--0.4，或对 priority 做分位数上限；
3. 更快将 beta 退火到 1，降低早期采样偏差；
4. 增大 replay warmup，先建立较完整的数据分布再启动 bootstrap；
5. 单独记录 positive/negative terminal transition 在 replay、priority top 1% 和 sampled
   batch 中的比例。

如果高 epsilon 下仍出现 Q 爆炸，再测试 learning rate `1e-4 -> 5e-5`、soft target update
或 clipped/ensemble Q；这些属于稳定性保护，不应替代 reward 学习信号诊断。

#### P3：增强 terminal credit，而不是继续放大 scale

n=3 对 24-step episode 仍然很短。可以比较 n=6/n=12，或者在 episode 完成后为整条轨迹
反向计算 Monte Carlo/terminal-aware return，使正向 terminal improvement 直接影响更早动作。
同时可采用按 reward 符号分层的 terminal replay，保证罕见的正向力学样本不会被大量随机
失败轨迹和 50k capacity 快速覆盖。

#### P4：补全状态并限制循环动作

当前 `prevent_revisit_positions=false`，观测也不含 visited mask 和 remaining horizon。
应优先加入：

```text
remaining_step_fraction
visited_position_mask
initial-to-current collision/H-bond/RMSD summary
initial/current predicted mechanical proxy
```

并禁止同一位置重复修改或立即 reversal。terminal predictor 基于结构 topology，而 Q state
主要是序列 ESM2 embedding；不补足结构/history 条件，增加探索只会产生更多相互冲突的
TD target。

#### P5：确认 reward predictor 对 strength 的局部灵敏度

在继续 RL 前，离线随机抽取蛋白与可接受 mutation，统计随机森林 strength/toughness delta
的零值率、唯一值数、分位数、单步/多步响应和模型不确定性。如果 strength 对局部 mutation
几乎不响应，RL 无法从 epsilon、PER 或更多 epoch 中创造缺失的监督信号。日志中还持续出现
RF 由 scikit-learn 1.6.1 保存、在 1.7.2 加载的版本警告，应先统一版本或重新训练 artifact，
保证 reward 可复现。

### 最终结论

慢化 epsilon 衰减的实验目标合理，但当前“所有 actor 同步进行四个 epoch 的高随机探索”
并不适合本项目。它显著加重 off-policy bootstrap 与 PER 的不稳定性，没有在相同数据预算下
提高 terminal strength/toughness；观察到的训练 total reward 上升主要由 epsilon 自身下降
造成。当前优先级不再是继续训练或进一步延长探索，而是：

```text
固定 greedy validation
-> actor-specific/受约束探索
-> PER 异常 priority 控制
-> 更长 terminal credit assignment
-> Markov state 与循环动作修复
-> reward predictor 局部灵敏度验证
```

只有固定验证集证明 checkpoint 的机械属性增量持续改善，才值得重新启动长时完整训练。

## 2026-08-19：固定 greedy validation、PER terminal 诊断与状态补全

### 1. 停止无效的慢 epsilon 衰减实验

通过目标 run directory 的 `train.pid` 在宿主机核对了此前使用
`--epsilon-decay-steps 739296` 的进程。PID `1678150` 的完整命令行与指定实验一致，随后向其
进程组 PGID `1678146` 发送 `SIGTERM`。进程在短暂清理后正常退出；再次检查该进程组为空，
没有残留 PyRosetta actor 或 ESM2 worker，也没有使用 `SIGKILL`。

`train_version_terminal_async.sh` 已删除显式的 `--epsilon-decay-steps 739296`，恢复 CLI 默认值
`50000` environment transitions。新一轮实验不再把四个 epoch 用于近似随机探索。

### 2. 固定 greedy validation

异步训练此前遇到 `--validate-every > 0` 时只输出“异步模式暂不支持”的 warning。本次增加了
一个专用 validation actor：

1. 从 validation index 的固定顺序选取前 16 个 PDB；
2. 启动时对随机初始化策略做一次 baseline validation；
3. 此后每完成 240 个训练 episode 执行一次；
4. 每个 PDB 固定使用 `validation_seed + validation_index`，当前 seed 为 `20260819`；
5. 每轮开始时冻结一份 online Q network 快照，整轮使用 `epsilon=0`；
6. validation actor 使用独立任务/策略队列，仅共享批量 ESM2 服务；
7. validation transition 不写入 replay，不推进 environment step，也不改变 epsilon schedule。

每个 PDB 的配对结果写入：

```text
<run_dir>/logs/validation_episodes.jsonl
```

聚合结果写入：

```text
<run_dir>/logs/validation_summary.jsonl
```

对 `terminal_reward`、`strength_delta_z` 和 `toughness_delta_z` 分别记录：

- mean；
- median；
- positive fraction；
- top 5% mean；
- top 10% mean；
- mean 的 bootstrap 95% confidence interval（1000 次重采样）。

这里 top-k 表示该目标增量最高的 k% validation 蛋白的平均增量，不是分类准确率。所有数值同时
写入 TensorBoard 的 `validation/*` tag，横轴使用冻结策略时的 environment step。固定 greedy
曲线才是判断策略是否真正改善力学性能的主要依据，训练 actor 的带 epsilon episode reward
仅作为 behavior-policy 诊断。

### 3. PER positive/negative terminal transition 诊断

ReplayBuffer format 从 v2 升级为 v3，每条 replay row 可额外保存：

```text
terminal_reward
terminal_strength_delta
terminal_toughness_delta
```

n-step accumulator 会把 episode 末端的三个标签传播到包含该 terminal outcome 的 n-step
prefix。旧 v1/v2 snapshot 仍可读取，缺失标签以 NaN 表示。

训练按 `--log-every-steps` 周期统计以下三个范围：

```text
per/replay/*
per/priority_top_1pct/*
per/sampled_batch/*
```

每个范围包含 terminal transition 占全部 transition 的比例，以及 terminal reward、strength、
toughness 的 positive/negative/zero fraction、样本数和均值。这可以直接判断 PER top 1% 与
实际 sampled batch 是否持续富集“大 TD error 但 toughness 严重下降”的轨迹。指标被写入
`optimization.jsonl` 和 TensorBoard 的 `optimization/per/*`。

### 4. 避免动作循环并补全 observation

启动脚本已同时打开：

```bash
--prevent-revisit-positions
--include-visited-mask-in-observation
```

action mask 会硬屏蔽本 episode 已修改的位置；环境还会在每个残基的 1280 维 ESM2 embedding
后追加一个 visited 标量通道，因此 Q head 输入为 `(L, 1281)`。初始值均为 0，某位置接受突变
后对应值变为 1。DDQNAgent、QNetwork、CPU actor policy、variable-length replay collation 和
checkpoint shape 检查均已适配该额外通道。

该实现解决了此前“约束存在于 action mask，但 Q observation 看不到 episode 历史”的部分
可观测问题。旧 `(L, 1280)` 模型仍受支持；新 `(L, 1281)` 运行不能直接恢复旧 checkpoint，
应启动新实验。

### 5. Replay capacity

脚本将 replay capacity 从 50,000 调整为 200,000，可覆盖约一个训练 epoch 的 transition。
当前 replay 保存 current/next ESM2 embedding，因此 200k 的 RAM 上限约为 50k 配置的四倍。
脚本继续使用 `--replay-checkpoint-every 0 --no-save-final-replay`，避免生成超大 replay 文件；
正式运行时仍需监控主机内存，若发生内存压力，应优先实现 embedding 去重/压缩，而不是静默
降低实际 capacity。

### 6. 启动参数与查看方式

```bash
bash train_version_terminal_async.sh
tensorboard --logdir <run_dir>/tensorboard
```

当前关键参数为：F8/G1、24 个 training actors、1 个 validation actor、PER、n-step=3、
step reward scale=0.025、terminal reward scale=8、replay capacity=200k、默认 50k epsilon
衰减、每 240 episodes 在 16 个固定 PDB 上做 greedy validation。

### 7. 测试结果

执行了脚本语法检查、Python 编译检查和完整测试套件：

```text
bash -n train_version_terminal_async.sh
python -m pytest -q
137 passed, 2 skipped in 10.19s
```

新增测试覆盖 visited observation 更新、1281 维 per-residue agent 前向、terminal metadata 的
n-step 传播、replay/top-1%/sampled-batch 三类 PER 统计、固定验证 bootstrap/top-k 汇总以及
validation TensorBoard 写入。两个 skipped 测试为原项目中依赖可选外部运行条件的测试，不是
本次修改失败。

## 2026-08-20：孤儿 worker 清理与 learner 生命周期保护

### 1. 宿主机孤儿进程清理

在宿主机进程命名空间中，以用户、`PPID`、`PGID` 和完整命令行联合核验，发现以下 5 个已经
失去 learner 的异步训练进程组：

```text
14691
719784
847883
872961
3071407
```

组内进程全部属于 `jianquanzhao`，`PPID=1`，命令均为 mprl-vgpt 环境中的
`multiprocessing.spawn` 或 `multiprocessing.resource_tracker`。对应的进程组 leader/learner
均已不存在，因此它们不是仍在运行的有效训练任务。5 个进程组合计占用超过 100 GiB RSS，
也是新任务发生主机内存压力的重要背景因素。

向 5 个进程组发送 `SIGTERM` 后，所有进程均在等待窗口内正常退出，不需要 `SIGKILL`。随后
再次扫描宿主机，未发现属于该用户且符合上述命令特征的 `PPID=1` 孤儿 worker。

### 2. Worker 父进程存活监控

此前 actor 和 ESM2 worker 只检查共享 `stop_event`。learner 正常进入 `finally` 时会设置该
事件；但 learner 被 OOM killer、管理员或外部程序直接 `SIGKILL` 后，没有存活进程负责设置
事件，worker 就会继续阻塞在任务、推理和结果队列上。

`model/asynchronous_module/runtime.py` 新增 `WorkerParentGuard`：

1. learner 在创建 worker 时将自己的 PID 显式传入；
2. worker 启动时验证实际 `PPID` 与 learner PID 一致；
3. Linux 下通过 `prctl(PR_SET_PDEATHSIG, SIGTERM)` 注册 parent-death signal；
4. 注册后再次检查 `PPID`，关闭“检查完成但 signal 尚未注册”这一竞态窗口；
5. ESM2 队列轮询、动态 batching、推理边界及 actor 任务轮询、环境 step 边界均执行 PPID
   检查，作为非 Linux 平台的回退机制，也提供明确的生命周期约束；
6. 父进程消失属于预期的异常关闭路径，worker 直接退出，不再尝试向已经无人消费的 fatal
   queue 写入消息。

该保护覆盖 training actor、固定 greedy validation actor 和全部 ESM2 inference worker。
正常训练结束仍沿用原来的 `stop_event -> sentinel -> join -> terminate` 清理流程。

### 3. Learner 退出码记录

新增可执行脚本：

```text
scripts/supervise_training.sh
```

`train_version_terminal_async.sh` 现在由低内存 supervisor 启动并等待 learner。运行目录新增：

```text
supervisor.pid
train.pid
learner_started_at
learner_finished_at
learner_exit_code
learner_exit_reason
```

`train.pid` 仍保存真实 learner PID，因此原有停止和检查方式保持兼容；`supervisor.pid` 单独记录
监督进程。正常结束会得到 `learner_exit_code=0`、`learner_exit_reason=completed`；如果 learner
被 `SIGKILL`，典型记录为 `137` 和 `signal:KILL`。supervisor 收到 `TERM/INT/HUP` 时会先把
信号转发给 learner，再等待并记录最终状态。

### 4. 验证结果

完成以下验证：

```text
bash -n train_version_terminal_async.sh scripts/supervise_training.sh
python -m py_compile asynchronous_training.py model/asynchronous_module/runtime.py
python -m pytest -q --disable-warnings
139 passed, 2 skipped in 8.69s
```

额外进行了两个生命周期集成测试：

1. supervisor 管理的短任务正常退出，记录 `0/completed`；
2. 对受 supervisor 管理的 learner 发送 `SIGKILL`，记录 `137/signal:KILL`；
3. 对已注册 parent-death signal 的测试 worker 杀死其父进程，worker 在 2 秒检查窗口内自动
   消失，没有形成 `PPID=1` 孤儿。

这次修改解决的是 learner 异常死亡后的可观测性与 worker 泄漏。它不会消除 learner 本身的
OOM 风险；当前 200k replay capacity 对 per-residue current/next ESM2 embedding 仍然非常
激进，后续仍应通过降低 capacity 或压缩、去重 replay state 控制峰值内存。

## 2026-09-10：正向终端样本保留与分层优先经验回放

### 1. 修改依据

截至上一轮长期训练快照，训练行为策略的 reward 已由较差状态明显“少负化”，但固定 greedy
validation 仍未稳定转正。Replay 诊断显示：终端 transition 约占全部 replay 的 12.5%，其中
正 terminal reward 约占 43.5%，所以真正的正向终端 transition 只占全部 replay 的约：

```text
12.5% * 43.5% = 5.4%
```

而 priority top 1% 中虽然终端 transition 占约 79%，正 terminal reward 只占约 11.5%。
这说明纯 absolute TD-error PER 主要优先学习“大幅失败”，有利于减少严重 toughness 下降，
但不会主动保证正向力学改善轨迹进入 batch。为此，本次没有取消 PER，而是在其上加入正向
终端样本的存储保留和分层采样。

### 2. 正向样本定义

正向 transition 定义为：

```text
is_positive = isfinite(terminal_reward)
              and terminal_reward > positive_reward_threshold
```

默认阈值为0。普通 transition 即使局部 step shaping 为正，只要没有携带 episode terminal
outcome，也不会进入正向层。当前使用 n-step=3，因此一个正向 episode 的 terminal outcome
会传播到最后3个 n-step prefix；这些 transition 均属于正向层。这一口径直接对应最终力学
代理改善，不会把局部碰撞或氢键 shaping 误当成终端成功。

### 3. 正向样本保留

`ReplayBuffer` 新增：

```text
positive_replay_reserve_fraction
positive_reward_threshold
positive_count
positive_fraction
```

buffer 未满时仍按原顺序写入，不人为丢弃数据。buffer 满载且已积累足够正样本后，负样本写入
会优先覆盖非正向 slot，避免正向样本比例跌破 reserve；正样本不足 reserve 时，新增正样本也
优先覆盖非正向 slot。该策略不复制 state/next_state，因此不会因正向保留再次放大 ESM2
embedding 的内存分配。

启动脚本默认：

```text
positive_replay_reserve_fraction = 0.10
positive_reward_threshold = 0.0
```

相对上一轮约5.4%的自然正向 transition 比例，10%的保留下限约提高至1.8倍。该下限只有在
数据流中实际发现足够正向 transition 后才能达到；它不能凭空制造正样本。

### 4. 正向分层 PER

采样时把 replay 划分为 positive 和 remaining 两个互斥层。batch 首先按
`positive_sample_fraction` 分配最低正向配额，然后在每层内部继续使用原 Ape-X 风格的
absolute TD-error priority：

```text
P(i | stratum) proportional to (abs(TD error_i) + epsilon)^alpha
```

importance weight 校正每层内部的 PER 偏置，但不会消除人为设定的层间正向配额。这样即使
beta 退火到1，正向 transition 仍保持目标 batch 占比，而不是被完整校正回约5.4%的原始
replay 分布。若某一阶段尚无正向样本，采样自动回退到原 uniform/PER；若正样本数量不足且
采用无放回采样，则只使用实际可用数量并由另一层补足 batch。

启动脚本默认：

```text
positive_sample_fraction = 0.25
```

batch size=128 时至少抽取32条正向 transition。相对上一轮自然期望约7条，单批正向曝光率
约提高4.6倍。其余96条仍覆盖零/负 terminal transition 和普通中间 transition，因此保留了
失败规避与 Bellman 状态覆盖。

### 5. 参数和启动方式

新增 CLI：

```text
--positive-sample-fraction
--positive-replay-reserve-fraction
--positive-reward-threshold
```

三个 CLI 参数默认分别为 `0/0/0`，因此普通 `training.py` 保持向后兼容。异步启动脚本
`train_version_terminal_async.sh` 显式采用 `0.25/0.10/0.0`，并支持环境变量覆盖：

```bash
MPRL_POSITIVE_SAMPLE_FRACTION=0.25 \
MPRL_POSITIVE_REPLAY_RESERVE_FRACTION=0.10 \
MPRL_POSITIVE_REWARD_THRESHOLD=0.0 \
bash train_version_terminal_async.sh
```

建议至少做以下三个同 environment-step、同 seed 对照：

```text
A: sample=0.00, reserve=0.00  原始 TD-PER 基线
B: sample=0.25, reserve=0.00  只检验正向采样效率
C: sample=0.25, reserve=0.10  检验采样与长期保留的组合效果
```

不要直接从旧 replay snapshot 续训后声称完成严格对照；新的分层语义应从新 replay 开始。
可以使用相同初始 agent checkpoint，但三组实验必须使用相同 checkpoint 和数据顺序。

### 6. 日志与判断标准

`terminal_outcome_diagnostics()` 新增：

```text
positive_terminal_count
positive_terminal_fraction
positive_reward_threshold
```

这些字段会进入 `optimization.jsonl` 和 TensorBoard：

```text
optimization/per/replay/positive_terminal_fraction
optimization/per/priority_top_1pct/positive_terminal_fraction
optimization/per/sampled_batch/positive_terminal_fraction
```

首先检查 replay 是否逐渐达到10%、sampled batch 是否稳定达到25%；随后以固定 greedy
validation 的 terminal reward mean/median、positive fraction、strength/toughness delta 和
bootstrap CI 为效果标准。若采样占比达到目标而固定验证仍不改善，主要瓶颈更可能是正向动作
可达性、状态表征或 reward predictor 灵敏度，而不是正向样本曝光不足。

### 7. 修改文件

```text
model/replay_buffer_module/replay_buffer.py
model/replay_buffer_module/README_REPLAY_BUFFER.md
training.py
asynchronous_training.py
train_version_terminal_async.sh
tests/test_replay_buffer.py
tests/test_training_multi_gpu.py
```

Replay snapshot format 从v3升级到v4，保存正向采样、reserve和阈值配置；v1-v3仍可按原配置
加载。新增测试覆盖正向 batch 配额、无正样本回退、满载后的 reserve、诊断字段及 snapshot
恢复。

### 8. 测试结果

```text
Python compile: passed
bash syntax: passed
git diff --check: passed
targeted replay/agent/training: 81 passed, 2 skipped
full repository: 152 passed, 2 skipped in 20.18s
```

额外比例 smoke test 使用容量100、自然正样本率5%的流，连续写入500条 transition：

```text
replay_size=100
positive_count=10
positive_fraction=0.10
sampled_batch_size=128
sampled_positive_count=32
sampled_positive_fraction=0.25
```

结果符合配置。该改造解决的是正向样本保留和曝光不足，不代表模型必然得到正 reward；是否
有效仍由固定 greedy validation 的跨 checkpoint 趋势和多 seed 对照决定。

## 2026-09-14：基于下置信界和 episode 去重的正样本采样

### 1. 修改动机与上一版失效原因

上一版把 `terminal_reward > 0` 的每条 n-step transition 都视为独立正样本。这个口径有两个
明显风险：

1. 很小的正值可能只来自 PyRosetta repack/relax 波动或 predictor 误差，并不代表可重复的
   力学改善。
2. `n_step=3` 会把同一个 episode 的 terminal outcome 传播到最后3条 replay row。同一条
   成功轨迹因此可以在一个 batch 中占据多个正样本位置，表面上提高了正样本比例，却没有
   增加独立成功事件的信息量。

本次将正样本判据改为：

```text
positive_episode = isfinite(terminal_reward_lcb)
                   and terminal_reward_lcb > positive_reward_lower_bound
```

比较使用严格大于。`positive_reward_lower_bound` 默认是0，可作为最小改善/噪声边界调高。
如果环境提供了由重复松弛计算的 `terminal_reward_lcb`，ReplayBuffer 直接使用它；如果没有，
则回退到 point terminal reward。这个回退保证当前训练可运行，但不等价于已经估计了统计
置信区间。要获得真正的 LCB，仍需按后文方案重复 repack/relax 后计算。

### 2. Episode 级去重实现

每条 actor transition 现在携带全局 `episode_id`。n-step accumulator 会把 terminal reward、
terminal reward LCB、strength/toughness delta 和 episode ID 一起传播到 terminal-bearing
prefix。

正向分层采样时先按 episode ID 分组，每个正向 episode 只保留一条候选 replay row：

```text
representative(e) = argmax priority_i, i belongs to positive episode e
```

随后在 episode representatives 中继续执行原有的 TD-error PER。这使高 TD-error 的成功经验
仍被优先学习，同时保证一次 batch 的正向配额不会被同一 episode 的3条 n-step row 重复
占据。旧 replay checkpoint 没有 episode ID，因此按未知 ID 载入并维持旧行为，不伪造
episode 边界。

当 unique positive episode 不足目标配额时，只抽取实际可用数量，由 remaining stratum 补齐；
当无放回 batch 在极端情况下连“唯一正 episode + remaining rows”都不足时，才回退到
transition 级采样以保证 learner 不因无法构造 batch 而停止。

### 3. 参数与兼容性

训练 CLI：

```text
--positive-sample-fraction       默认 0.25
--positive-reward-lower-bound    默认 0.0
--positive-replay-reserve-fraction
```

`--positive-reward-threshold` 作为旧名称仍然可用，并映射到同一个参数。异步启动脚本提供：

```text
MPRL_POSITIVE_SAMPLE_FRACTION
MPRL_POSITIVE_REWARD_LOWER_BOUND
MPRL_POSITIVE_REPLAY_RESERVE_FRACTION
```

旧环境变量 `MPRL_POSITIVE_REWARD_THRESHOLD` 也保留为 lower-bound 的回退值。Replay snapshot
格式升级到 v5，新增 `terminal_reward_lcbs` 和 `episode_ids`；v1-v4 仍可读取，旧版本 LCB
回退为 point terminal reward、episode ID 记为未知。

建议顺序运行三组同 seed、同初始 agent、同数据顺序、同 environment-step 的对照，不要让三组
同时竞争 PyRosetta CPU 和 ESM GPU：

```bash
MPRL_POSITIVE_SAMPLE_FRACTION=0.10 \
MPRL_POSITIVE_REWARD_LOWER_BOUND=0.0 \
MPRL_RUN_LABEL=positive_episode_10pct \
bash train_version_terminal_async.sh

MPRL_POSITIVE_SAMPLE_FRACTION=0.15 \
MPRL_POSITIVE_REWARD_LOWER_BOUND=0.0 \
MPRL_RUN_LABEL=positive_episode_15pct \
bash train_version_terminal_async.sh

MPRL_POSITIVE_SAMPLE_FRACTION=0.25 \
MPRL_POSITIVE_REWARD_LOWER_BOUND=0.0 \
MPRL_RUN_LABEL=positive_episode_25pct \
bash train_version_terminal_async.sh
```

第一轮先固定 lower bound 为0，只比较采样比例。第二轮应使用 predictor 重复性实验得到的噪声
阈值或经验 LCB，避免同时改变两个变量。

### 4. 新增诊断与判断标准

`terminal_outcome_diagnostics()` 的正样本统计改为基于 LCB，并新增：

```text
terminal_reward_lcb_*
known_episode_transition_count
unique_episode_count
episode_duplicate_fraction
positive_known_episode_transition_count
positive_unique_episode_count
positive_episode_duplicate_fraction
positive_reward_lower_bound
```

这些字段会沿用现有记录链进入 `optimization.jsonl` 和 TensorBoard 的以下命名空间：

```text
optimization/per/replay/*
optimization/per/priority_top_1pct/*
optimization/per/sampled_batch/*
```

正常情况下，`sampled_batch/positive_terminal_fraction` 应接近设置的10%、15%或25%，且
`sampled_batch/positive_episode_duplicate_fraction` 应为0。最终效果仍以固定 greedy
validation 的 terminal reward、strength/toughness delta、positive fraction 和 bootstrap CI
为准，不能以 sampled batch 的正样本占比作为模型改善证据。

### 5. 力学 predictor 可执行分析方案

#### 5.1 要检验的假设

```text
H1 稀疏性：在一个给定结构上，真正改善力学性能的单点突变占比极低。
H2 不可重复性：突变效应小于 repack/relax 和 predictor 的波动，reward 符号不稳定。
H3 不可识别性：改善动作客观存在，但当前 ESM2 sequence state 无法预测其方向。
H4 OOD/代理失真：突变结构离 predictor 训练分布过远，RF 外推结果不能作为可靠排序。
```

#### 5.2 固定扫描面板

从固定 validation split 中选择24个蛋白，按 sequence length、初始 strength、初始 toughness
分层抽样，并固定 PDB 列表和随机种子。该面板不能参与 predictor 或辅助 probe 的训练。
每个蛋白最多均匀抽64个可变位置，短蛋白使用全部位置；每个位置扫描除 wild type 外的19种
氨基酸。pilot 上限约为：

```text
24 proteins * 64 positions * 19 substitutions = 29,184 actions
```

第一阶段每个 action 只执行一次与 RL 完全一致的 local repack/relax 和七特征 RF 推理，记录：

```text
protein_id, position, wt_aa, mutant_aa, seed
strength_initial, strength_final, delta_z_strength
toughness_initial, toughness_final, delta_z_toughness
terminal_reward_point
7 initial features, 7 final features, feature deltas
Rosetta energy, accepted/rejected, failure reason
```

#### 5.3 重复性与 LCB

每个蛋白从初筛结果选择 top 20、接近0的20个和 bottom 20个 action，分别用5个固定但不同的
PyRosetta seeds 从同一个初始结构重新执行 repack/relax。每个 action 计算 reward mean、SD、
符号一致率，以及单侧95% bootstrap LCB：

```text
LCB_95(action) = percentile_5%(bootstrap means)
```

同时估计 action 间方差与同一 action 重复方差，并计算：

```text
ICC = variance_between_actions
      / (variance_between_actions + variance_within_action)
```

这一步给出可用于训练的实际 `terminal_reward_lcb`，也给出合理的
`--positive-reward-lower-bound`。RF 各树的标准差可以作为 OOD 辅助指标，但不能替代重复
结构松弛的经验置信区间。

#### 5.4 稀疏性和可达上界

分别以 point reward 和 `LCB_95 > lower_bound` 统计：

```text
positive action density per protein
至少存在1/5/10个正向 action 的蛋白比例
top-1、top-5、top-10 attainable terminal reward
strength/toughness 同时改善的 Pareto-positive 比例
按位置、二级结构、氢键网络区域和氨基酸替换类型分层的正动作密度
```

除总体均值外必须报告 protein-level median 和 bootstrap CI，防止少数长蛋白凭借更多 action
主导统计结果。

#### 5.5 当前 observation 的动作可识别性

在 protein-grouped cross-validation 下训练两个只用于诊断的轻量 probe：

1. `ESM2 per-residue embedding + position + mutant-AA one-hot -> action reward/positive LCB`；
2. `初始结构七特征 + candidate feature delta -> action reward/positive LCB`，作为结构信息上界。

比较 held-out protein 上的 Spearman、top-5%/top-10% hit rate、enrichment factor、PR-AUC 和
校准曲线。若结构 probe 明显有效而 ESM2 probe 无效，说明当前 RL state 存在部分可观测性，
应把氢键/拓扑/局部几何特征加入 observation，或在 action 选择时加入廉价结构 look-ahead；
此时继续提高 replay 正样本比例不会解决根因。

#### 5.6 OOD 检查与决策门槛

把 candidate 的七维特征与 RF 训练特征分布比较，记录 robust z-score、最近邻距离和 RF tree
dispersion。建议用以下门槛作为下一步工程决策，而不是当作生物学定律：

```text
稀疏：median positive-LCB action density < 1%，或 >50% 蛋白没有正向 action
噪声主导：median sign agreement < 0.8，或 ICC < 0.5
状态不可识别：ESM2 probe Spearman < 0.2 且 top-10% EF < 1.5，结构 probe 明显更高
明显 OOD：正向候选主要集中在训练特征范围之外，且 tree dispersion 同时升高
```

对应策略：可靠但稀疏时改进 actor proposal/curriculum；噪声主导时使用重复松弛、ensemble 和
LCB reward；状态不可识别时补充结构 observation；OOD 或代理平坦时先重训/校准 mechanical
predictor，再继续 RL。

#### 5.7 推荐实现与产物

后续实现建议采用 CPU PyRosetta worker 并行、主进程集中写表，避免每个 worker 同时写 CSV：

```text
code:
model/reward_module/mechanical-properties-predictor/analyze_reward_landscape.py

outputs:
outputs/mechanical_property_predictor/reward_landscape/
  panel.csv
  single_mutation_screen.parquet
  repeated_relax.parquet
  protein_summary.csv
  probe_metrics.json
  figures/
```

建议 CLI 形态：

```bash
python model/reward_module/mechanical-properties-predictor/analyze_reward_landscape.py \
  --pdb-index <fixed_validation_index> \
  --model-artifact params/hbond_random_forest.joblib \
  --num-proteins 24 --max-positions 64 \
  --screen-repeats 1 --confirm-repeats 5 \
  --bootstrap-samples 2000 --lcb-alpha 0.05 \
  --workers 24 --seed 20260914 \
  --output-dir outputs/mechanical_property_predictor/reward_landscape
```

先完成约29k action 的 pilot 和约7.2k次确认重复，再依据结果决定是否扩展到全部位置或更多
蛋白；直接对约7k个蛋白做 `L*19*5` 全扫描成本过高，且在判断 H1-H4 前没有必要。

### 6. 修改文件与验证

```text
model/replay_buffer_module/replay_buffer.py
model/replay_buffer_module/n_step.py
model/replay_buffer_module/__init__.py
model/replay_buffer_module/README_REPLAY_BUFFER.md
model/asynchronous_module/runtime.py
training.py
asynchronous_training.py
train_version_terminal_async.sh
tests/test_replay_buffer.py
tests/test_n_step.py
tests/test_training_multi_gpu.py
```

测试结果：

```text
Python compile: passed
bash syntax: passed
git diff --check: passed
targeted replay/n-step/training/asynchronous: 62 passed, 2 skipped
full repository: 156 passed, 2 skipped in 12.02s

batch-size 128 ratio smoke test:
10% -> 13 positive rows from 13 unique positive episodes
15% -> 20 positive rows from 20 unique positive episodes
25% -> 32 positive rows from 32 unique positive episodes
```

本轮没有启动长期 RL，也没有凭 RF tree dispersion 伪造 LCB。代码已经具备接收真实
`terminal_reward_lcb` 的接口；应先通过上述固定面板测出 reward landscape 和重复松弛噪声，
再决定 lower bound 与10%/15%/25%中哪一个采样比例值得进入长期训练。

## 2026-09-15：ESM2 + PyRosetta 正向动作监督预训练方案评估

### 1. 总体结论

该方案合理，且比继续单独调整 PER 正样本比例更接近当前问题的核心：PER 只能增加已经发现的
成功经验的学习次数，不能提高 actor 在巨大动作空间中首次发现成功动作的概率；监督预训练可以
直接给 Q head 一个“哪些突变更可能改善力学性能”的初始排序。

但是不建议把方案实现为“只收集正样本，然后把正样本 action 做普通分类 SFT”。只有正样本
没有同一状态下的负样本和近零样本，模型无法学习动作之间的相对优劣，也容易把所有动作分数
同时抬高。更合适的定义是：

```text
基于结构扫描结果的 supervised action-value / action-ranking pretraining
```

它可以视为面向 DDQN 的 SFT，也与 demonstration pretraining 的思想一致。监督阶段学习动作
排序和保守的一步改善值，随后再由 DDQN 学习多步 long-horizon return。

建议采用以下完整路线：

```text
固定训练蛋白和结构状态
    -> ESM2 编码当前序列
    -> 枚举/提议候选突变
    -> 从同一个 current pose 独立执行 PyRosetta local repack/relax
    -> mechanical predictor 计算 paired reward mean 和 LCB
    -> 构造 positive + hard-negative + neutral 的状态级排序数据
    -> 监督预训练当前 per-residue Q head
    -> 从中间多突变状态继续扫描并聚合数据
    -> online/target network 同步加载 SFT checkpoint
    -> DDQN 在线微调
```

### 2. ESM2 在方案中的准确职责

“使用 ESM2 单独探索正样本”需要稍作修正。ESM2 给出的是序列表征和进化/语言模型意义上的
氨基酸合理性，并不直接知道氢键拓扑或力学性能，因此不能单独判断 mechanical-positive
action。推荐分工如下：

1. ESM2 per-residue embedding 是 SFT/DDQN 的状态输入；
2. ESM2 masked-token probability 可以作为候选突变 proposal 或结构合理性过滤条件；
3. PyRosetta 负责构造突变后的局部结构；
4. 当前七特征 random forest predictor 负责产生 strength/toughness 和 scalar reward 标签；
5. 重复 repack/relax 的 paired bootstrap LCB 负责判断该正向标签是否超过噪声。

如果对每个位置的19种替换全部计算 reward，候选生成并不需要 ESM2；这时 ESM2 只需对每个
唯一 state 编码一次。若后续为了节省计算只保留 ESM2 top-k substitution，必须先在全扫描
pilot 上检查它对 positive-LCB action 的 recall，避免进化合理性过滤掉罕见但有效的力学突变。

### 3. 该方案能解决和不能解决的问题

能够缓解：

```text
冷启动：随机 Q head 在 L*19 级动作空间中没有有效排序。
奖励稀疏：模型在 RL 前已经接触可靠的正向和负向动作对比。
正样本曝光不足：一个可靠正向 action 可直接监督，而不必等待在线 actor 偶然访问。
训练初期 Q 排序混乱：SFT 可提高 greedy/top-k proposal 的正动作密度。
```

不能自动解决：

```text
reward predictor 本身不准确或被优化利用；
PyRosetta 重复松弛噪声大于突变效应；
ESM2 sequence observation 无法识别依赖当前三维构象的动作；
单点改善不具有可加性，多个突变存在显著 epistasis；
只扫描初始 WT，而 RL 在第2至24步访问的是完全不同的多突变状态。
```

因此 SFT 是否值得扩展，必须由正在进行的 LCB/action-landscape 实验先证明“可靠正动作确实
存在”，再由 held-out protein 的监督排序结果证明“当前 observation 能识别这些动作”。

### 4. 数据生成方案

#### 4.1 数据边界和拆分

只对 RL training split 生成 SFT 数据。固定 greedy validation PDB 以及与其高度相似的序列
cluster 必须完全排除，防止 SFT 预先看过验证动作。

当前索引约有7700个 training PDB 和855个 validation PDB；力学标签 CSV 中可解析到7041条
序列，平均长度105.39 aa。按 CSV 粗略计算，完整单点全扫描规模为：

```text
sum(sequence_length * 19) = 14,099,102 candidate actions
```

训练索引与 CSV 数量并不完全一致，因此正式扫描前需要生成一个 manifest，只保留 PDB、序列、
predictor 标签和 split 能唯一匹配的 entry，并记录所有被排除原因。

直接对全部数据执行约1410万次 PyRosetta 更新并不适合作为第一步。按此前异步训练约10个
environment step/s 的量级粗估，一次扫描就需要约16天，重复5次则不可接受。建议分三级执行：

```text
Pilot：24个分层蛋白，最多64个位置，每个位置19个替换，约29,184 actions。
Scale-1：512至1000个 sequence-cluster 分层蛋白，先单次扫描，再确认候选。
Scale-2：只有 Pilot/Scale-1 证明有效后，才扩展蛋白数量或位置覆盖。
```

#### 4.2 每个 state 的扫描规则

对一个给定 current state：

1. 缓存一次 ESM2 embedding 和 action mask；
2. 对每个允许位置生成19个非 no-op substitutions；
3. 每个候选都从完全相同的 current pose clone 开始，禁止前一个候选影响后一个候选；
4. 使用与 RL 相同的 local repack/relax、radius、score function 和 predictor artifact；
5. 保存失败、拒绝和结构质量异常的候选，不能只保存成功结果；
6. 对初筛 top、near-zero、bottom 以及随机候选执行多个固定 seed 的重复松弛；
7. 使用配对差值减少初始构象噪声。

一个中间状态 `s` 上 action `a` 的监督标签应同时保留：

```text
R_absolute(s+a) = terminal score of candidate relative to episode initial structure
R_marginal(s,a) = terminal score(s+a) - terminal score(s)
```

动作排序主要使用 `R_marginal`，因为它回答“在当前状态继续执行该突变是否改善”；
`R_absolute` 用于检查整个 episode 是否已经达到正 terminal reward。重复松弛时对每个 seed
先计算 paired marginal difference，再对 paired means bootstrap，得到：

```text
LCB_95(s,a) = percentile_5%(bootstrap paired mean reward)
positive(s,a) = LCB_95(s,a) > configured_noise_boundary
```

#### 4.3 必须包含的样本类型

不能只保存 positive action。每个 state 至少包含：

```text
confirmed positive：LCB 超过边界；
hard negative：point reward 看起来为正，但 LCB 不大于边界；
near-zero：处于 predictor/repack 噪声带内；
clear negative：稳定降低 terminal reward；
structural failure：碰撞、缺失原子、repack/relax 失败或明显 OOD。
```

hard negative 尤其重要，它直接教模型不要把偶然的 point-positive 当作机械改善。batch 应按
state/protein 平衡，而不是让长蛋白或正样本多的蛋白贡献更多权重。

推荐数据字段：

```text
protein_id, sequence_cluster, state_id, parent_state_id, mutation_depth
sequence, visited_mask, current_pdb, state_hash
position, wildtype_aa, mutant_aa, action_index, valid_action
reward_point, reward_mean, reward_sd, reward_lcb, reward_sign_agreement
delta_strength_mean/lcb, delta_toughness_mean/lcb
initial/final 7 structural features and feature deltas
Rosetta energy delta, predictor OOD diagnostics, seed, failure reason
```

ESM2 embedding 应按 `state_hash + model_version` 缓存为 float16/memmap，数据表只保存 cache key，
不要为每个 action 重复存储同一份 `(L,1280)` embedding。

### 5. 从单点 WT 扩展到多步状态

只在 WT 上扫描会造成明显 covariate shift：SFT 学到第一步后，RL 的后续23步仍处于未见状态。
不建议穷举多步组合，而采用迭代式数据聚合：

```text
Round 0：扫描 WT states，训练 SFT-v0。
Round 1：用 SFT-v0 在 training proteins 上走到 depth 1/2/4，保存访问状态并扫描候选。
Round 2：训练 SFT-v1，再收集 depth 4/8/16 状态。
Round 3：检查 depth 24 的固定 greedy rollout，不再默认扩大数据。
```

中间状态不需要扫描全部动作。可组合以下候选：

```text
当前 SFT top-32 actions
ESM2 plausibility top-16 actions
uniform/chemically-diverse random 16 actions
```

这样既覆盖模型认为好的动作，也保留发现模型盲区的机会。每轮必须按 `state_hash` 去重，并保留
visited mask；相同序列但已访问位置不同是不同的 RL state。

### 6. 监督目标：排序优先，不直接拟合长程 Q

当前网络接收 contextual ESM2 per-residue embedding 加 visited flag，并对每个 residue 输出20个
Q values。其结构可以直接用于监督预训练，无需先更换网络。

推荐每个 state 构造一个保守目标分布：

```text
y(s,a) = clipped reward_lcb(s,a)
p*(a|s) = softmax(y(s,a) / temperature), valid actions only
```

第一版损失建议由三部分组成：

```text
L_listwise = KL[p*(a|s) || softmax(Q(s,a)/temperature_q)]
L_rank     = max(0, margin - Q(s,positive) + Q(s,hard_negative))
L_reg      = Huber(Q(s,a), clipped reward_lcb(s,a))

L_SFT = L_listwise + lambda_rank * L_rank + lambda_reg * L_reg
```

`L_listwise` 学习整个 action surface 的相对排序；`L_rank` 强化正样本与难负样本的间隔；
`L_reg` 约束输出尺度，避免所有 Q values 同时增大。lambda、temperature 和 margin 应由 SFT
validation 调节，而不应先固定成未经验证的常数。

保留 strength 和 toughness 的独立标签用于分析和可选 auxiliary loss，但基本版仍使用与 RL
一致的 `0.5 * delta_z_strength + 0.5 * delta_z_toughness` scalar label。这样不会在 SFT 和 RL
之间偷偷改变优化目标。

这里拟合的是 conservative one-step action improvement，不是真正的24步 discounted Q return。
所以 SFT checkpoint 提供的是动作排序先验，最终 Q calibration 仍由 DDQN TD learning 完成。

### 7. SFT 训练、验证和进入 RL 的门槛

SFT 数据必须按 sequence-similarity cluster 做 train/validation/test split，不能按 action row
随机切分。否则同一个蛋白不同位置会同时进入训练和测试，指标会严重虚高。

held-out proteins 上至少记录：

```text
Spearman(Q, reward_lcb)
positive@1 / positive@5 / positive@10
top-5% and top-10% hit rate
enrichment factor and NDCG
greedy regret = max_a reward_lcb(s,a) - reward_lcb(s,argmax Q)
greedy action reward mean/median and bootstrap CI
strength/toughness delta and Pareto-positive fraction
```

建议满足以下条件后再进入长期 RL：

```text
greedy positive@1 的 bootstrap lower CI 高于 random-action baseline；
top-10% enrichment factor 至少明显大于1，并以2作为有价值的初始目标；
greedy selected action 的 mean reward LCB 不为负；
收益能在未见 sequence clusters 上复现，而不是只在 SFT train proteins 上出现。
```

若 SFT train loss 很低但这些 held-out 指标无效，说明问题不是 RL 探索技巧，而是 reward
不可重复、状态不可识别或数据泄漏；此时不应继续扩大 SFT 数据。

### 8. SFT checkpoint 接入 DDQN

SFT checkpoint 必须记录 ESM2 model/version、embedding dimension、hidden dims、visited-mask
配置、action mapping、reward scale 和数据 manifest hash。接入时：

1. 使用相同的 `QNetwork` 配置实例化 online network；
2. 加载 SFT Q-head 参数；
3. 将 online 参数完整复制到 target network；
4. 新建 optimizer，不加载 SFT optimizer momentum；
5. environment/optimization step 从0开始；
6. 前若干 optimizer steps 使用较低 learning rate 或 warmup，防止 TD loss 立即抹掉排序先验；
7. 可在早期保留一个逐渐衰减的 ranking auxiliary loss，随后完全交给 DDQN。

第一轮对照只改变初始化：

```text
A：random Q initialization + current DDQN
B：SFT Q initialization + current DDQN，其他参数全部相同
```

只有 B 在固定 environment steps 下显著优于 A 后，才增加 demonstration replay 或 guided
exploration，避免无法判断收益究竟来自 SFT、采样还是 epsilon 策略。

第二轮可测试：

```text
C：SFT initialization + 5%至10% high-confidence demonstration replay
D：SFT initialization + annealed SFT proposal / online-Q mixture
```

demonstration 必须是 episode/state 去重后的 positive-LCB 数据，并同时保留 hard negatives；不能
重新退化为上一版“重复抽取 point-positive terminal rows”的策略。

### 9. 在线评估设计

四组实验使用完全相同的 protein order、PyRosetta seeds、environment-step budget 和固定 greedy
validation：

```text
random-init DDQN
SFT only（不做 RL，用于测量监督策略上界）
SFT-init DDQN
SFT-init DDQN + demonstration/auxiliary loss（第二阶段）
```

主要指标不是 training loss，而是：

```text
首次达到 positive validation mean/median 的 environment steps；
validation terminal reward vs environment steps 的 AUC；
positive fraction、strength/toughness mean/median 和 bootstrap CI；
不同 mutation depth 上的 reward trajectory；
动作多样性、重复位置率和 predictor OOD fraction。
```

至少运行3个 training seeds。只有 SFT-init 在 held-out fixed greedy validation 上更早达到正值、
最终 CI 更高，才能说明它缓解了奖励稀疏，而不是仅让训练 replay reward 看起来更好。

### 10. 主要风险及对应控制

```text
只优化 RF 代理：加入 feature-range/OOD 约束，并人工检查 top candidates。
结构噪声：用 paired repeated relax LCB，而不是 point reward。
只会第一步：通过多轮中间状态数据聚合解决。
模型只记蛋白：sequence-cluster split，按 protein/state 等权。
只学正样本：加入 hard negative、near-zero、failure 和 listwise ranking。
ESM2 过滤漏掉机械突变：先在全19替换 pilot 上测 positive recall。
Q 尺度与 long-horizon return 不一致：Huber/clip 控制尺度，随后由 TD 微调校准。
SFT 被在线训练快速遗忘：optimizer 重置、LR warmup、短期衰减 auxiliary ranking loss。
```

当前 Q head 只显式接收 ESM2 sequence embedding 和 visited flag，而 mechanical predictor 依赖
氢键/拓扑结构特征。如果结构 probe 能识别正动作、ESM2 probe 不能，下一版模型应把当前结构的
七个归一化特征、当前 predicted strength/toughness、remaining horizon 等作为全局 context
broadcast 到 residue head。SFT 不能从输入中恢复根本不存在且与序列不唯一对应的信息。

### 11. 推荐执行顺序

```text
1. 等待当前 repeated-relax LCB pilot，确认 positive-LCB density 和重复性。
2. 在同一24蛋白面板上训练最小 action-ranking probe，验证 ESM2 observation 的可识别性。
3. 若 probe 有效，扩展至512个 sequence-cluster 分层蛋白并训练 SFT-v0。
4. 用 held-out cluster 的 positive@k、EF、regret 和实际 PyRosetta greedy rollout 验收。
5. 通过验收后收集 depth 1/2/4/8 中间状态，训练 SFT-v1。
6. 先做 random-init DDQN vs SFT-init DDQN 的单变量对照。
7. 只有 SFT 初始化有效但在线遗忘明显时，再引入 demonstration replay/auxiliary ranking。
```

最终判断：该方案值得做，但决定成败的不是“正样本数量足够多”，而是正动作是否具有统计可重复
性、是否在蛋白级独立测试集上可排序、以及训练数据是否覆盖 RL 真正访问的多突变状态。按上述
门槛分阶段推进，可以在投入约1410万次全量结构计算前尽早识别方案是否成立。

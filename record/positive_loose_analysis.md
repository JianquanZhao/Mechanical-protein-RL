# Positive loose replay training analysis

Analysis date: 2026-10-12

## 1. Scope

The target run is:

```text
async_a24_esm8_f8_g1_prioritized_possample0.25_poslcb-0.25_
posreserve0.10_n3_stepx0.025_terminalx8.0_20260917_100006
```

It completed all 492,864 episodes (64 epochs), from 2026-09-17 10:00 to
2026-10-07 16:21, with learner exit code 0. The main configuration was:

```text
actors                         24
train frequency / grad steps  8 / 1
replay                         TD-prioritized, capacity 100,000
positive sample fraction      0.25
positive replay reserve       0.10
positive threshold            -0.25
n-step                         3
step reward scale             0.025
terminal reward scale         8.0
max episode steps             24
epsilon                       1.0 -> 0.05 in 50,000 environment steps
```

Two controls were used:

1. `20260915_145850`: positive sample fraction 0 and reserve 0. This is the
   closest no-positive-quota control, but it was killed after 42,000 episodes.
2. `20260910_110708`: the older `terminal_reward > 0` positive replay scheme,
   stopped after 122,000 episodes.

All three runs otherwise share the important environment, reward, model,
PER, n-step and update-schedule settings. Comparisons are made at matched
episode/global-step horizons first; the 492,864-episode result is then analyzed
separately.

## 2. Executive conclusion

The replay mechanism worked technically, but the current experiment did not
demonstrate that high-confidence positive replay solves the sparse-reward problem.

1. The 25% quota was reached: in the 0.9M-1.0M environment-step window, qualified
   terminal rows were 10.01% of replay and exactly 25.00% of sampled batches.
2. At the common 42,000-episode horizon, the new run was worse than the no-quota
   control in greedy terminal reward and toughness. It therefore did not improve
   early sample efficiency.
3. Across the complete run, the fixed validation reward improved from clearly
   negative to less negative. The first-50 versus last-50 validation comparison
   improved terminal reward by +0.285 and toughness by +0.0697; 13/16 proteins
   improved on both metrics. Strength did not improve detectably.
4. The final objective was still not positive: the last 50 greedy validations had
   mean terminal reward -0.234, median -0.068, and positive fraction 39.4%.
   No validation run had a bootstrap lower confidence bound above zero.
5. Most importantly, this run did not contain a separately calculated relaxation
   LCB. Missing `terminal_reward_lcb` falls back to point `terminal_reward`, and
   the threshold was -0.25. The actual experiment is therefore
   `point terminal reward > -0.25 + episode-deduplicated quota`, not
   `repeated-relaxation LCB > 0`.

The defensible conclusion is: the run produced a real late-stage improvement,
mainly by reducing toughness degradation, but it does not establish that the LCB
data strategy caused that improvement or that the reward-sparsity problem is solved.

## 3. Did the replay distribution change as intended?

Yes. The following values are means from the matched 0.9M-1.0M environment-step
window:

| Diagnostic | New loose threshold | No quota | Old reward > 0 |
|---|---:|---:|---:|
| Replay qualified-terminal fraction | 10.01% | 6.64% | 10.01% |
| Sampled qualified-terminal fraction | 25.00% | 11.99% | 25.00% |
| Sampled terminal-transition fraction | 40.60% | 29.59% | 43.26% |
| Sampled terminal reward mean | -0.252 | -0.531 | -0.106 |
| Sampled toughness delta mean | -0.057 | -0.112 | -0.030 |
| Sampled positive episode duplication | 0.00% | 0.57% | not logged |

Episode deduplication worked at sampling time: positive samples in a batch came
from unique episodes. In replay storage, positive episode duplication remained
about 66.7%, which is expected from `n_step=3`: the terminal outcome is propagated
to the final three n-step rows. The 10% reserve is transition-level rather than
unique-episode-level.

However, satisfying the quota did not make the batch mechanically positive. A
threshold of -0.25 explicitly treats mildly negative terminal outcomes as qualified.
The remaining 75% is still selected by absolute TD priority. In the same window,
the priority top 1% had:

```text
qualified-terminal fraction   1.58%
terminal reward mean         -2.130
toughness delta mean         -0.397
```

Thus the learner received more useful terminal examples than the no-quota run,
but also continued to receive a strong stream of severe negative-TD examples.
The net sampled reward remained negative.

## 4. Matched 42,000-episode comparison

The no-quota run only reached 42,000 episodes, so this is the fairest direct
comparison. Values below are averages over the final 20 greedy validations before
42,000 episodes:

| Greedy validation metric | New loose threshold | No quota | Old reward > 0 |
|---|---:|---:|---:|
| Terminal reward mean | -0.534 | -0.454 | -0.528 |
| Terminal positive fraction | 30.9% | 31.9% | 30.3% |
| Strength delta mean | -0.0493 | -0.0566 | -0.0573 |
| Toughness delta mean | -0.0841 | -0.0568 | -0.0747 |

Across all 176 paired validation checkpoints up to 42,000 episodes, new minus
no-quota was:

```text
terminal reward mean difference  -0.125
moving-block 95% interval         [-0.196, -0.062]
toughness mean difference         -0.0291
moving-block 95% interval         [-0.0386, -0.0205]
strength mean difference          -0.0021
moving-block 95% interval         [-0.0119,  0.0071]
```

The exploratory training episodes give the same direction in their final 5,000
episodes:

| Training metric | New loose threshold | No quota | Old reward > 0 |
|---|---:|---:|---:|
| Total reward mean | -0.453 | -0.385 | -0.432 |
| Terminal reward mean | -0.356 | -0.287 | -0.335 |
| Terminal-positive fraction | 35.1% | 38.3% | 36.4% |
| Strength+toughness Pareto-positive | 19.3% | 22.3% | 20.9% |

Therefore, the loose positive quota did not provide an early learning advantage.
Relative to no quota, its principal early deficit was toughness.

## 5. Comparison with the old positive definition

At approximately 121,000 episodes, the final 20 validation means were:

| Metric | New loose threshold | Old reward > 0 |
|---|---:|---:|
| Terminal reward | -0.499 | -0.709 |
| Strength delta | -0.0423 | -0.0736 |
| Toughness delta | -0.0823 | -0.1035 |

The per-protein late-window comparison favored the new run for terminal reward
on 12/16 proteins. New minus old terminal reward was +0.210 with a protein-bootstrap
95% interval `[+0.026, +0.466]`; strength was +0.0313 `[+0.0040, +0.0720]`.
Toughness was +0.0212, but its interval crossed zero.

This is encouraging but not definitive. Across all matched checkpoints to 121,000
episodes, the terminal-reward difference was only +0.0008 with an interval that
crossed zero. The late advantage emerged after initially different trajectories,
and the runs did not start from an identical saved Q-network state.

## 6. Full-run effect

The target run is the only one of these three experiments that completed 64 epochs.
Its fixed greedy validation trajectory was:

| Episode range | Terminal mean | Positive fraction | Strength mean | Toughness mean |
|---|---:|---:|---:|---:|
| 0-50k | -0.556 | 26.8% | -0.0427 | -0.0962 |
| 100k-150k | -0.511 | 29.2% | -0.0416 | -0.0863 |
| 150k-200k | -0.386 | 31.7% | -0.0291 | -0.0672 |
| 250k-300k | -0.544 | 30.4% | -0.0474 | -0.0886 |
| 300k-350k | -0.444 | 31.1% | -0.0404 | -0.0704 |
| 400k-450k | -0.309 | 38.2% | -0.0336 | -0.0437 |
| 450k-end | -0.322 | 36.3% | -0.0367 | -0.0438 |

The trajectory is noisy and non-monotonic. A sustained improvement only becomes
clear after about 300k-400k episodes (roughly epochs 39-52), far beyond both
controls. Consequently, the late gain cannot be separated from simply training
for much longer.

Using each of the 16 fixed proteins as the paired unit, first 50 versus last 50
validation runs gives:

| Metric | Mean change | Protein-bootstrap 95% interval | Improved proteins |
|---|---:|---:|---:|
| Terminal reward | +0.285 | [+0.031, +0.539] | 13/16 |
| Toughness delta | +0.0697 | [+0.0287, +0.1122] | 13/16 |
| Strength delta | +0.0015 | [-0.0301, +0.0387] | 8/16 |

This shows that the full-run improvement is not produced by only one or two
validation proteins. It is specifically a reduction in toughness degradation;
there is no evidence of a strength gain.

Absolute final performance remains insufficient. Over the last 50 validations:

```text
terminal reward mean                  -0.234
terminal reward median                -0.068
terminal-positive fraction             39.4%
validation runs with mean reward > 0    4.0%
validation runs with 95% CI low > 0     0.0%
strength delta mean                   -0.0289
toughness delta mean                  -0.0296
```

The last 10,000 exploratory training episodes were also still negative on average:

```text
total reward mean / median       -0.136 / -0.089
terminal reward mean / median    -0.063 / -0.019
total-positive fraction           44.1%
terminal-positive fraction        48.7%
Pareto-positive fraction          28.4%
```

The result is therefore accurately described as “from worse to less bad”, not as
reliable mechanical-property optimization above the wild type.

## 7. Optimization behavior

The new sampling scheme did not improve early numerical stability relative to the
no-quota control. Up to one million environment steps, sampled p99 values were:

| Metric | New loose threshold | No quota | Old reward > 0 |
|---|---:|---:|---:|
| Loss | 2.07 | 1.05 | 1.83 |
| Absolute TD error | 6.44 | 3.83 | 6.41 |
| Grad norm | 3.09 | 2.04 | 3.30 |
| Mean Q value | 12.91 | 10.32 | 13.32 |

All runs later entered low-median regimes. Nevertheless, the final 100,000 sampled
optimization steps of the complete run still had median Q=0.331 and median TD
error=0.575, while p99 Q=16.89 and p99 TD error=11.07. This is a heavy-tailed,
intermittently unstable regime rather than clean uniform convergence. It completed
without NaN/Inf failure, but the replay mixture still periodically exposes the
network to very large targets.

## 8. Why this was not a true LCB experiment

`terminal_outcome_fields()` reads `info["terminal_reward_lcb"]`, but when the
environment does not provide it the code assigns:

```python
terminal_reward_lcb = terminal_reward
```

The replay buffer repeats the same fallback. The current environment terminal
reward path writes `terminal_reward` and predictor metrics, but does not run
multiple independent repack/relax replicas or publish a separately estimated LCB.
In the optimization diagnostics, `terminal_reward_lcb_mean` and
`terminal_reward_mean` are identical throughout this run.

In addition, the configured lower bound is -0.25, not 0. A sample can lose
mechanical reward and still enter the positive quota. With equal strength/toughness
weights and terminal scale 8, this permits approximately
`delta_strength_z + delta_toughness_z > -0.0625`. The term “qualified loose sample”
is therefore more accurate than “high-confidence positive sample”.

## 9. Experimental limitations

1. The no-quota control ended at 42k episodes and the old-positive run ended at
   122k; neither tests the 300k-493k interval where the new run improved most.
2. There is only one run per setting. Time-series checkpoints are autocorrelated
   and are not independent random seeds.
3. `DDQNAgent` seeds its NumPy action RNG but does not call `torch.manual_seed`
   before creating `QNetwork`. The three runs therefore began from different
   network weights despite `seed=7`.
4. `env.reset(seed=...)` seeds NumPy/action-space sampling, but PyRosetta was
   initialized only with `-mute all`; stochastic repacking is not explicitly
   given a constant Rosetta seed. This is consistent with different validation
   values at global step 0 despite the same validation PDBs and validation seed.
5. The 16-protein validation set is useful for paired monitoring but too small to
   support a broad generalization claim.

These limitations do not erase the observed late improvement, but they prevent a
causal claim that the loose positive replay strategy produced it.

## 10. Recommended next experiment

Before changing more optimizer hyperparameters, the data-side hypothesis should
be tested in its intended form:

1. Implement genuine repeated local repack/relax measurements and publish both
   point reward and `terminal_reward_lcb`. Assert in logs that they are not
   identically equal.
2. Define a qualified positive as `reward_lcb > noise_floor`, initially with
   `noise_floor=0`, rather than using -0.25. Add a toughness non-degradation
   constraint, or require Pareto-positive strength/toughness LCBs, because the
   current failure mode is dominated by toughness loss.
3. Seed PyTorch before network construction and use a controlled PyRosetta seed.
   Save one initial Q-network checkpoint and start every replay ablation from that
   exact checkpoint.
4. Compare no quota, 10%, 15% and 25% quota using at least three seeds and the same
   environment-step budget. The no-quota control must run into the 400k-episode
   region before it can test whether the late improvement is strategy-specific.
5. Report the point-reward/LCB disagreement rate, unique positive episodes in
   replay, sampled Pareto-positive fraction, and fixed-validation paired deltas.
   Do not use the achieved 25% batch quota itself as evidence of policy improvement.
6. Consider a stratified mixture of confirmed positive, near-zero, hard negative
   and ordinary PER rows. The current remaining 75% PER mass is dominated by
   catastrophic negative TD errors, so increasing one quota alone does not define
   a balanced reward landscape.

The present run is a useful baseline: it proves that replay composition can be
controlled and that long training can reduce toughness degradation. The next
decisive test is a reproducible, genuine-LCB ablation; if that still cannot move
greedy validation above zero, SFT of the Q head on explicitly mapped action
landscapes is the more promising next stage.

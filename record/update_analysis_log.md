# Training Log Analysis: `/tmp/mprl-tensorboard-folder-smoke/logs/`

分析时间：2026-06-23  
日志目录：

```text
/tmp/mprl-tensorboard-folder-smoke/logs/
```

包含文件：

```text
episodes.jsonl
episodes.csv
steps.jsonl
optimization.jsonl
validation.jsonl
```

## 1. 总体结论

这次训练没有出现 NaN、Inf、程序级崩溃或明显的环境 step 中断，因此“训练流程可以跑通”。但从指标看，这还不是一个健康收敛的训练结果，主要问题集中在：

1. reward 几乎完全被 `collision` 惩罚主导；
2. 所有完整 episode 的总 reward 都是负数；
3. Q value 在训练后半程快速升高，存在 Q overestimation / Q scale drift 风险；
4. `grad_norm` 持续升高，后期有梯度变大的趋势；
5. terminal reward 始终为 0，说明当前训练主要只在优化 step reward；
6. 日志中存在一个未完成的 partial episode，说明这次 run 可能被中途停止或最后一次 episode 没完整写入 episode summary。

如果这是一次 smoke / tensorboard 测试，这个结果是可以接受的：流程、日志、ReplayBuffer、优化器都在工作。  
如果这是一次正式训练，目前还不能说明策略已经学到了稳定有效的突变策略。

## 2. 日志规模与一致性

当前快照统计：

```text
episodes.jsonl:      85 complete episode records, episode 0..84
steps.jsonl:         8525 step records
optimization.jsonl:  7509 optimization records
validation.jsonl:    170 validation records
```

完整 episode 记录：

```text
episode_steps = 100 for all 85 complete episodes
terminal_reward = 0.0 for all 85 complete episodes
unique source_pdb = 14
planned_episodes = 10240
```

一致性问题：

```text
steps.jsonl contains episode 85 with only 25 steps.
episodes.jsonl ends at episode 84.
```

说明：

- episode 85 已经开始写 step records；
- 但没有对应的 `episodes.jsonl` summary；
- 这通常意味着训练被中途停止，或最后一个 episode 还没结束时进程退出；
- 因此 episode 级统计主要参考 episode 0..84，step/optimization 统计包含 episode 85 的 partial records。

建议：

- 正式分析时优先使用完整 episode；
- 如果要自动分析，应过滤掉 `steps.jsonl` 中没有对应 episode summary 的 partial episode；
- 可以在训练脚本中增加 graceful shutdown，收到中断信号时写出 partial episode summary。

## 3. 训练是否稳定

### 3.1 数值稳定性

未发现 NaN / Inf：

```text
episodes:      no non-finite numeric values
steps:         no non-finite numeric values
optimization:  no non-finite numeric values
validation:    no non-finite numeric values
```

这是好信号，说明当前 reward、ReplayBuffer、DDQN loss 和 optimizer 至少没有数值崩坏。

### 3.2 Episode reward

完整 episode 的 total reward：

```text
n=85
min    = -8224
p25    = -2510
median = -1618
mean   = -2006
p75    = -1220
max    = -397.7
```

前 10 个 episode 平均：

```text
-2533.86
```

后 10 个完整 episode 平均：

```text
-1335.24
```

判断：

- 后期 reward 比前期明显没那么差，这是一个正向迹象；
- 但所有 episode 仍为负值，说明策略还没有找到整体正收益的突变路径；
- 当前更像是“训练开始有改善趋势”，不是“策略已经有效收敛”。

### 3.3 Step reward

step reward：

```text
n=8508+
min    = -1937
p25    = -5.859
median = -0.666
mean   = -20.06
p75    = -0.0048
max    = 2.5
```

正负 step 数：

```text
positive step rewards: 918
negative step rewards: 7590
```

判断：

- 大多数动作带来负 reward；
- 少量动作能得到正 reward，但正收益上限很小，最大约 `2.5`；
- 负收益极端值很大，最差 step 约 `-1937`；
- reward 分布明显长尾，训练目标会被少数 collision 极端惩罚主导。

### 3.4 Optimization loss / TD error

Optimization records：

```text
loss:
  min    = 2.315
  median = 16.39
  mean   = 19.58
  max    = 87.39

mean_absolute_td_error:
  min    = 2.763
  median = 16.83
  mean   = 20.02
  max    = 87.86
```

判断：

- loss 和 TD error 没有爆炸到不可控；
- 但 TD error 均值约 20，仍偏高；
- 后期 loss 没有明显下降到低水平，说明 Q function 尚未稳定拟合当前 replay 分布。

## 4. 环境是否正常

### 4.1 Episode 长度

所有完整 episode 都是：

```text
episode_steps = 100
```

step 终止分布：

```text
normal non-terminal steps: 8423
truncated terminal steps: 85
terminated terminal steps: 0
```

判断：

- 环境没有提前因为无合法动作而频繁结束；
- action mask 没有明显导致动作空间过早耗尽；
- episode 都是由 `max_steps=100` 截断结束；
- 从运行稳定性看，环境 step 链路是正常的。

### 4.2 Terminal reward

所有完整 episode：

```text
terminal_reward = 0.0
```

判断：

- 当前训练实际没有利用终端力学性能预测目标；
- 策略优化几乎完全依赖 step reward；
- 如果项目目标是增强力学性能，那么后续必须接入真实 terminal predictor，或明确当前阶段只做结构稳定性预训练。

### 4.3 Reward components

累计 reward components：

```text
collision:       -169762.30
backbone_hbond:     -246.00
sidechain_hbond:    -573.50
local_rmsd:          -91.30
```

判断：

- `collision` 是绝对主导项；
- hbond 和 local RMSD 对总 reward 的影响很小；
- 当前 reward 实际上近似为 collision penalty optimizer；
- 如果这是预期的结构稳定性训练，可以接受；
- 如果想让策略学习“保留/增强氢键、控制局部结构漂移、提升力学性能”，当前 reward 权重或 scale 需要重新校准。

### 4.4 Collision 问题

collision step 统计：

```text
reward_component/collision:
  median = -0.2466
  mean   = -19.93
  min    = -1937.06
```

极端负 step 基本来自 collision：

```text
worst step reward ~= -1936.57
collision term    ~= -1937.06
```

判断：

- 有少数突变导致严重 clash；
- 这些严重 clash 强烈拉低 episode reward；
- repack/minimize 并没有完全消除这些局部冲突；
- 可能需要：
  - 调整 mutation 后 repack/minimize 参数；
  - 限制高风险突变动作；
  - 对 collision reward 做 clipping / robust scaling；
  - 或在 environment 中对严重 clash 的 candidate 直接 reject，而不是 commit。

## 5. 策略是否有问题

### 5.1 Epsilon

episode epsilon：

```text
max = 0.9981
min = 0.8385
```

optimization epsilon：

```text
first optimization epsilon = 0.981
last optimization epsilon  = 0.838348
```

判断：

- 当前仍处于高探索阶段；
- 策略还没有进入主要 exploitation 阶段；
- 因此不能用这次短 run 判断最终策略质量；
- reward 后期改善可能部分来自 Q 学习，也可能来自随机探索分布变化。

### 5.2 Q value drift / overestimation 风险

Optimization Q value：

```text
mean_q_value:
  min    = -9.142
  median = 17.75
  mean   = 25.58
  max    = 98.16

mean_target_q_value:
  min    = -85.35
  median = 2.985
  mean   = 9.01
  max    = 90.04
```

从 chunk trend 看：

```text
chunk 0 mean_q ~= -2.35
chunk 3 mean_q ~= 14.88
chunk 5 mean_q ~= 43.48
chunk 7 mean_q ~= 82.69
```

判断：

- Q value 在后半程快速变大；
- 但实际 episode reward 仍为负；
- 这提示 Q function 可能出现正向偏移或 overestimation；
- mean Q 和 mean target Q 都在变大，但 Q 的增长速度更明显；
- 建议密切观察更长训练中 `q_values.png` 是否继续单调飙升。

建议：

- 降低 learning rate；
- 缩短 target sync interval 或尝试 soft update；
- 对 reward 做 clipping / normalization；
- 增大 replay warmup 或 batch size；
- 观察 per-residue action mask 是否让 max-Q 偏向某些重复模式。

### 5.3 Gradient norm

grad norm：

```text
min    = 0.190
median = 9.792
mean   = 11.22
max    = 53.85
```

chunk trend：

```text
chunk 0 grad_mean ~= 2.66
chunk 3 grad_mean ~= 9.38
chunk 5 grad_mean ~= 15.49
chunk 7 grad_mean ~= 23.47
```

判断：

- 梯度范数随训练推进明显增大；
- 如果 `max_grad_norm=10.0`，当前日志中的 `grad_norm` 已经频繁超过该阈值；
- 需要确认日志中的 `grad_norm` 是 clipping 前还是 clipping 后。如果是 clipping 前，说明梯度经常被裁剪；如果是 clipping 后，说明裁剪没有按预期限制住。

建议：

- 确认 `grad_norm` 记录语义；
- 降低 `--learning-rate`；
- 尝试更强 reward scaling / clipping；
- 检查 collision 极端负样本是否造成 TD target 长尾。

## 6. Validation 结果

Validation records：

```text
n = 170
validation episodes = 85
validation indices = 0, 1

total_reward:
  min    = -4231
  p25    = -202.1
  median = -53.1
  mean   = -210.6
  p75    = -16.95
  max    = -0.058
```

Validation 前 10 个 episode 平均：

```text
-280.63
```

Validation 后 10 个 episode 平均：

```text
-179.85
```

判断：

- validation reward 比 training reward 好很多，但仍为负；
- validation 有改善趋势，但波动非常大；
- 因为只用了 2 个 validation PDB，结论不稳；
- validation reward 比 training reward 好，可能不是泛化更好，而是 validation PDB 更容易、长度/结构分布不同，或者训练 PDB 中有更严重 collision 长尾。

建议：

- 增加 `--validation-episodes`；
- 按 PDB 长度、初始 collision score、清洗比例分层看 validation；
- 记录 validation 的 reward components，而不只记录 total reward；
- 对 train 和 val 使用相同的 PDB 难度统计，避免误判泛化。

## 7. 数据 / PDB 层面问题

完整 episode 只覆盖：

```text
unique source_pdb = 14
```

但 `planned_episodes = 10240`，说明这只是一次很早期的 smoke / partial run。

按 PDB 的平均 episode reward 差异很大，例如：

```text
worst PDB mean reward ~= -3255
best  PDB mean reward ~= -1092
```

判断：

- 不同 PDB 难度差异明显；
- 一些 PDB 会系统性产生更大的 collision penalty；
- 当前训练效果高度受 PDB 分布影响；
- 后续正式训练应按 `source_pdb` 聚合 reward、collision、local_rmsd、accepted/error 信息。

建议：

- 找出最差 PDB，单独检查初始结构、长度、清洗情况、collision score；
- 对 source PDB 做 blacklist / quarantine 机制；
- 或在 dataset preprocessing 中增加初始 Rosetta score / fa_rep 阈值。

## 8. 主要问题清单

### 问题 1：Reward 被 collision 单项支配

证据：

```text
collision cumulative contribution: -169762
local_rmsd cumulative contribution: -91
backbone_hbond cumulative contribution: -246
sidechain_hbond cumulative contribution: -573.5
```

影响：

- agent 主要学习避免 collision；
- hbond 和 local RMSD 几乎不会影响策略；
- 力学目标尚未进入训练信号。

建议：

- 重新标定 reward scales；
- 对 collision 做 clipping；
- 或将 severe collision candidate 作为 invalid transition / reject。

### 问题 2：Q value 后期快速升高

证据：

```text
mean_q_value from ~0 to ~98
actual episode rewards remain negative
```

影响：

- 存在 Q overestimation 或 reward scale drift 风险；
- 如果继续训练，可能出现策略过度相信某些动作。

建议：

- 继续观察 `q_values.png`；
- 降低 learning rate；
- 调整 target sync；
- 进行 reward normalization。

### 问题 3：Gradient norm 持续升高

证据：

```text
grad_norm mean: 11.22
grad_norm max:  53.85
late chunk grad_mean ~= 23.47
```

影响：

- 优化过程后期变得更不稳定；
- 与 collision 长尾和 Q value drift 可能相关。

建议：

- 检查 grad_norm 是 clipping 前还是 clipping 后；
- 降低 learning rate；
- 加强 reward clipping；
- 增大 batch 或 replay warmup。

### 问题 4：Terminal reward 没有发挥作用

证据：

```text
terminal_reward = 0.0 for all complete episodes
```

影响：

- 当前并没有直接优化最终力学性能；
- 训练目标仍是结构稳定性 proxy。

建议：

- 接入真实 mechanical predictor；
- 或明确当前阶段是 pretraining / stability-only base run。

### 问题 5：日志存在 partial episode

证据：

```text
steps.jsonl has episode 85 with 25 steps
episodes.jsonl ends at episode 84
```

影响：

- 自动分析时可能把不完整 episode 混入 step-level 统计；
- episode-level 和 step-level 数量不严格一致。

建议：

- 分析时过滤 partial episode；
- 训练脚本增加 graceful shutdown；
- 或在 logger 中写入 `run_status` / `episode_finalized` 字段。

## 9. 下一轮训练建议

建议下一轮正式训练前优先做这些调整：

1. 降低 reward 长尾影响：

```text
对 collision_loss_used 做 clipping 或 robust normalization。
```

2. 增加 validation 覆盖：

```text
--validation-episodes 10 或更高
```

3. 观察 Q drift：

```text
重点看 q_values.png、td_error.png、grad_norm.png。
```

4. 如果 Q 和 grad 继续升高：

```text
降低 --learning-rate
增大 --replay-warmup-size
考虑缩短 --target-sync-interval
```

5. 按 PDB 聚合诊断：

```text
source_pdb -> mean total_reward
source_pdb -> mean collision_loss_used
source_pdb -> worst step reward
```

6. 如果目标是力学性能增强：

```text
尽快接入 terminal mechanical predictor，否则当前训练只是在做 step-level structural proxy optimization。
```

## 10. 最终判断

### 训练

训练没有数值崩溃，可以继续跑更长实验；但 Q value 和 grad_norm 后期升高，需要警惕优化不稳定。

### 环境

环境基本正常：episode 都能跑满 100 step，未出现大量提前终止或 action mask 耗尽。但 reward 显示 mutation/repack/minimize 后仍有严重 collision 长尾。

### 策略

策略目前还不能认为有效。reward 后期有改善迹象，但仍全部为负；epsilon 仍很高，探索占主导；Q value 的乐观偏移和实际负 reward 不匹配，需要进一步调参和更长训练验证。

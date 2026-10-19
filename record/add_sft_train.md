# Mechanical-protein DDQN supervised pretraining plan

## 2026-09-16：认知校正、数据生成解耦与 A-D 实验方案

### 1. 对当前认知的评估

整体方向正确，而且已经形成了合理的实验主线：先使用结构计算得到动作监督信号，对当前 Q head
进行监督预训练，再比较 random initialization 与不同 SFT 使用方式对 DDQN 的影响。以下认识
可以直接保留：

1. A-D 四组策略能够区分随机初始化、SFT 初始化、demonstration replay 和 guided proposal 的
   独立贡献；
2. SFT 数据只能来自允许用于训练的蛋白，必须按序列相似度 cluster 做 group split；
3. 只使用正样本是不充分的，必须同时保留负样本、近零样本和结构失败样本；
4. 数据生成与模型训练解耦是正确的工程选择，PyRosetta-heavy 阶段适合放到 CPU 服务器；
5. SFT 既要看 regression metrics，也要看 state 内 action-ranking metrics；
6. 数据标签必须复用 RL 的结构处理、力学 predictor、氨基酸 action mapping 和 reward 定义。

以下几点需要校正或进一步明确。

#### 1.1 当前模型并不是联合训练 ESM2 + MLP

当前 `ESM2SequenceEncoder` 使用 `model.eval()` 和 `torch.inference_mode()`，ESM2 是冻结的
observation encoder。DDQN 实际优化的是接收 `(L, 1280+visited_flag)` observation、逐 residue
输出20个 action values 的 MLP Q head。

因此基础版本的 SFT 应定义为：

```text
冻结 ESM2，监督预训练当前 per-residue Q head
```

不建议第一版微调 ESM2。否则数据量、显存、checkpoint、异步 actor 推理和模型广播都会一起
改变，无法判断收益来自 action supervision 还是语言模型微调。未来若需要 LoRA/adapter，应作为
独立实验。

#### 1.2 CPU 数据生成不应强制加载 ESM2

如果每个位置扫描全部19种非 wild-type 氨基酸，CPU 数据生成阶段不需要 ESM2。它只需要输出
sequence、visited mask、action、结构结果和 reward labels；SFT 训练时再在 GPU 上批量计算或读取
缓存的 ESM2 embeddings。

如果要用 ESM2 masked-token probability 缩小候选集，应把 proposal 生成做成一个独立的 GPU
预处理任务，输出候选 action manifest，再交给 CPU PyRosetta workers。这样 CPU 服务器无需
安装 GPU 环境，也不会因 ESM2 CPU inference 成为新瓶颈。

#### 1.3 “与 RL 相同的一切 score”需要区分主目标和辅助信息

当前目标是提高 mechanical terminal reward，而不是重新让局部 step shaping 主导策略。因此
建议：

```text
SFT primary target:
    paired mechanical marginal reward LCB

SFT auxiliary records/targets:
    strength delta, toughness delta, point terminal reward,
    step reward and its components, Rosetta energy, update failure/OOD flags
```

可以完整记录 RL 的所有 score，但不应默认把 step reward 与 terminal reward 相加后作为唯一 SFT
标签。否则可能重新学到“避免局部碰撞”而不是“改善力学性能”。如需拟合 exact one-step RL
reward，应作为 auxiliary/ablation，并保持当前 `step_reward_scale` 与 `terminal_reward_scale`
完全一致。

#### 1.4 单点扫描样本不能直接等同于 demonstration replay

相互独立的单点扫描非常适合监督 action ranking，但它们不天然构成满足当前 MDP 和 n-step
return 的完整 episode。C 方案中的 demonstration replay 应在 SFT 模型训练后，使用固定策略
实际运行完整轨迹，记录正确的 state、next_state、done、n-step return 和 terminal outcome。

否则将“从相同 WT 独立出发的候选”错误串成 episode，会产生错误 Bellman target。

### 2. 推荐的总体代码和项目边界

不建议让 `training.py` 在每次 RL 启动时先运行 SFT。数据生成、SFT 训练和 RL 应是三个可独立
复现和恢复的阶段：

```text
Stage 1: CPU data generation repository/tool
         PDB -> mutation scan -> PyRosetta -> reward replicates -> LCB dataset

Stage 2: SFT training entry point
         sequence/state + action labels -> frozen ESM2 -> pretrained Q-head checkpoint

Stage 3: existing DDQN training entry point
         optional Q initialization / demo replay / proposal mixture
```

推荐的数据生成项目结构：

```text
mechanical-protein-sft-data/
├── configs/
│   ├── scan_base.yaml
│   └── reward_contract.yaml
├── src/mechanical_sft_data/
│   ├── build_source_manifest.py
│   ├── cluster_and_split.py
│   ├── build_scan_tasks.py
│   ├── scan_worker.py
│   ├── merge_shards.py
│   ├── estimate_lcb.py
│   └── validate_dataset.py
├── scripts/
│   ├── run_cpu_scan.sh
│   └── resume_cpu_scan.sh
└── tests/
```

当前 RL 仓库增加：

```text
model/sft_module/
├── dataset.py
├── collate.py
├── losses.py
├── trainer.py
├── metrics.py
└── proposal_policy.py

train_sft.py
```

`training.py` 只增加加载和实验控制接口，不在内部生成 SFT 数据：

```text
--q-init-checkpoint
--q-init-lr-warmup-steps
--demonstration-data
--demonstration-sample-fraction
--sft-proposal-checkpoint
--sft-proposal-start-mix
--sft-proposal-end-mix
--sft-proposal-decay-steps
```

这里的 SFT optimizer warmup 属于 `train_sft.py`；加载 SFT 后的 RL learning-rate warmup 属于
`training.py`。两者应使用不同参数名，避免概念混淆。

### 3. 数据源、去重和 split

#### 3.1 两种可选的数据治理方式

为了与当前 DDQN 基线严格比较，短期推荐：

```text
现有 RL train index
    -> exact-sequence dedup
    -> sequence-similarity clustering
    -> cluster-level SFT train/validation/test split

现有 fixed RL validation/test
    -> 完全不参与 SFT 数据生成和超参数选择
```

这样 A-D 可以沿用相同的 RL 验证集和现有基线。

长期更严格的版本可以对全部蛋白先统一聚类，再重新生成 RL/SFT master split。但这会改变 RL
训练集和验证集，A 组 random-init baseline 也必须重新训练，不能直接与旧日志比较。

#### 3.2 去重不等于删除所有同序列结构

建议分两层处理：

1. 完全相同 sequence 的重复 PDB：选择一个 canonical structure 进入主扫描；
2. 同 sequence 的不同实验/预测构象：保留在 repeatability/OOD 子集中，用于估计结构不确定性，
   但不能跨 SFT train/validation/test split。

sequence cluster 必须作为不可拆分 group。可先使用 MMseqs2/CD-HIT 的 identity + coverage
阈值建立版本化 cluster manifest，再按 cluster 分配 80/10/10，同时对 sequence length、baseline
strength/toughness 和 topology bins 做近似分层。具体 identity threshold 应做敏感性检查，不应
只报告一个未经验证的阈值。

所有 split 统计都应同时报告 protein 数、cluster 数、总 residue 数和 action 数；不能只按 action
row 比例划分，因为长蛋白会造成数据量假平衡。

### 4. CPU 数据生成的详细流程

#### 4.1 冻结 reward contract

数据生成前先输出 `reward_contract.json`，至少包含：

```text
PyRosetta version and initialization flags
local repack radius, minimize/no-minimize, backbone settings
score function and all step-reward normalization parameters
terminal predictor artifact path + SHA256
the seven feature definitions and order
strength/toughness scalarization weights
step_reward_scale and terminal_reward_scale
amino-acid ordering and action-index mapping
missing-atom/data-cleaning rules
random seed list
source code git commit
```

当前 RL 启动脚本使用 `--no-minimize`。SFT 扫描必须显式继承这一配置；不能使用更充分的 relax
生成标签后，再让 RL 在另一种结构动力学下执行动作。

建议同时保存 raw physical prediction、raw normalized delta 和 scaled RL reward。这样未来修改
`terminal_reward_scale` 时不必重新运行全部 PyRosetta。

#### 4.2 构造 state 和 candidate action

每个扫描 state 必须包含：

```text
state_id
parent_state_id
source_protein_id
initial_episode_structure_id
current sequence
current pose checksum/path
mutation depth
visited-position mask
valid-action mask
```

对每个 valid position 枚举19个非 no-op amino acids。每个 candidate 都必须从相同 current pose
clone 开始，禁止按 action 顺序连续修改同一个 pose，否则标签会混入前一个候选的结构变化。

当前 environment 只在 episode 截断时自动计算 terminal reward。因此独立扫描程序不能只执行
一次 `env.step()` 后读取 `info["terminal_reward"]`；在 `max_steps>1` 时该值通常为0。扫描器应
显式调用 terminal calculator/finalization，并分别计算：

```text
R_absolute(s+a): candidate relative to original episode initial pose
R_marginal(s,a): candidate score - current state score
```

SFT action ranking 以 `R_marginal` 的 LCB 为主，`R_absolute` 用于判断整条设计是否已经达到正
terminal improvement。

#### 4.3 重复 repack/relax 与噪声边界

先验证 PyRosetta seed 的确改变 packer/relax 轨迹。如果重复运行结果完全相同，不能把相同输出
复制多次后声称得到置信区间。

建议先建立 null controls：对 WT/current pose 执行相同次数的 no-op repack 或等价结构流程，
估计“没有真实氨基酸变化时”predictor reward 的噪声分布。噪声边界可以由 null distribution
的高分位数确定，而不是主观设定。

candidate 与 current baseline 使用相同 seed 做 paired difference：

```text
d_r = R_after(seed_r) - R_before(seed_r)
```

重复次数采用顺序扩展：

```text
R=1：全候选初筛；
R=5：top、near-zero、bottom 和随机候选的最低确认级别；
R=10：最终 high-confidence label 的推荐起点；
R=20：置信区间仍跨越噪声边界的关键候选。
```

5个重复可用于 pilot，但 percentile bootstrap CI 很不稳定，不应作为最终 demonstration 的唯一
依据。保存所有 replicate-level rows，再由独立脚本计算 mean、SD、sign agreement、paired
bootstrap LCB/UCB，避免 worker 自己汇总后丢失原始信息。

建议标签定义：

```text
confirmed_positive:
    reward_lcb > positive_noise_boundary

hard_negative:
    point/mean reward > 0 but reward_lcb <= positive_noise_boundary

near_zero_or_uncertain:
    confidence interval overlaps the null noise band

confirmed_negative:
    reward_ucb < negative_noise_boundary

structural_failure:
    mutation/repack/feature/predictor failed or candidate is outside validity rules
```

#### 4.4 CPU 并行与可恢复性

PyRosetta 使用多进程而不是 Python threads。每个 worker 独立初始化 PyRosetta 并处理 deterministic
task shards。actor 数量必须通过吞吐和内存 benchmark 选择，不能默认等于 CPU 核数，因为每个
PyRosetta process 都会占用较多内存。

推荐先测1、8、16、24 workers 的：

```text
actions/s
RSS per worker and total RSS
failure rate
I/O wait
mean and p95 action latency
```

每个 shard 使用原子写入：先写临时文件，完成校验后 rename。task manifest 记录 pending、running、
complete、failed 和 retry count，使 CPU 作业可跨机器、跨日期恢复。worker 不并发追加同一个大
CSV；每个 shard 独立写 Parquet，最终由单进程 merge/validate。

只为 WT/current states、confirmed candidates、demonstration trajectories 和 debug failures 保存
PDB。为全部候选保存 PDB 会产生不必要的存储和 inode 压力。

#### 4.5 数据版本和跨项目契约

最终 dataset release 至少包含：

```text
dataset_version.json
source_manifest.parquet
cluster_manifest.parquet
split_manifest.parquet
states.parquet
actions.parquet
replicates/*.parquet
structures/selected/
validation_report.json
checksums.sha256
```

`dataset_version.json` 保存 source commit、reward contract checksum、predictor checksum、cluster
tool/version、seed list、统计计数和 schema version。当前 RL 仓库只依赖这个版本化数据契约，
不依赖数据生成项目的内部实现。

### 5. SFT dataset 组织

SFT 的基本训练单位应是一个 state 及其候选 action set，而不是彼此无关的 action rows：

```text
state observation: ESM2(sequence) + visited mask
valid mask:        L * 20
targets:           reward LCB/mean and strength/toughness deltas per valid action
quality weights:   repeat count, uncertainty, OOD/failure flags
```

如果一个 state 扫描了全部 valid actions，可以直接做 listwise learning；若只扫描部分候选，
必须提供 `label_available_mask`，不能把未扫描动作误当成 reward=0。

每个 state 必须包含：

```text
confirmed positives
hard negatives
near-zero/uncertain actions
clear negatives
structural failures or invalid actions
```

建议 state/protein/cluster 等权采样，再在 state 内平衡 action types。不要简单 oversample 所有
positive rows，否则一个长蛋白或一个特别容易优化的 state 会支配梯度。

ESM2 embedding 可以在 SFT GPU 节点上按 `state_hash + ESM model checksum` 计算并缓存。使用
float16/bfloat16 cache、memory mapping 和 variable-length collate，避免每个 action 复制同一份
`(L,1280)` embedding。

### 6. SFT 模型和损失

基础版本保持当前模型架构不变：

```text
frozen ESM2 per-residue embeddings
    + visited flag
    -> shared residue MLP
    -> 20 action scores per residue
```

主监督目标为 conservative marginal mechanical reward：

```text
y(s,a) = clip(reward_lcb_marginal(s,a), target_min, target_max)
p*(a|s) = softmax(y(s,a) / target_temperature), valid+observed actions only
```

推荐联合损失：

```text
L_listwise = KL[p*(a|s) || softmax(Q(s,a) / q_temperature)]
L_rank     = max(0, margin - Q(s,positive) + Q(s,hard_negative))
L_reg      = Huber(Q(s,a), y(s,a), sample_weight=confidence_weight)

L_SFT = L_listwise + lambda_rank * L_rank + lambda_reg * L_reg
```

`L_listwise` 学 state 内整体排序，`L_rank` 特别处理 positive/hard-negative，`L_reg` 约束 Q 输出
尺度。loss weights、temperature、margin 和 target clipping 由 SFT validation 选择，不能根据
test split 调参。

第一版保留 strength/toughness 独立标签用于 metrics；只有在 scalar ranking 明显被目标冲突限制
时，再增加两个 auxiliary heads/losses。这样先验证最小方案，避免一次改变模型架构和初始化方式。

SFT 使用的 reward 应保持 RL scale。如果为数值稳定对 target 做标准化，checkpoint 必须保存 scaler，
并在初始化 DDQN 前把输出层恢复到 RL reward scale；否则虽然排序正确，初始 Q magnitude 仍会与
TD target 不匹配。

### 7. SFT 训练参数和 checkpoint

建议 `train_sft.py` 支持：

```text
data/split:
--data-dir
--split-manifest
--train-split --val-split --test-split

model:
--esm-model-dir --embedding-dim
--hidden-dims
--include-visited-mask

optimization:
--epochs --state-batch-size
--learning-rate --weight-decay
--warmup-ratio/steps
--lr-scheduler
--gradient-clip
--early-stopping-patience
--seed

loss:
--listwise-weight --ranking-weight --regression-weight
--target-temperature --q-temperature
--ranking-margin
--target-clip-min --target-clip-max

runtime/logging:
--num-workers --device
--output-dir --enable-tensorboard
--resume-checkpoint
```

checkpoint 需要记录：

```text
Q-head state_dict
architecture and amino-acid action ordering
ESM checkpoint name/checksum and embedding dimension
visited-mask configuration
target scaling/clipping
reward-contract and SFT-dataset checksums
best validation metrics
optimizer/scheduler state for SFT resume
```

RL 只读取 Q-head 和兼容性 metadata，不继承 SFT optimizer momentum。

### 8. 指标定义和统计方式

#### 8.1 Regression metrics

```text
MAE
RMSE
R2
Pearson
global Spearman
```

同时报告 action-level micro average 和 protein/state-level macro average。由于每个 protein 的候选
数量不同，仅报告全体 action 的 global metric 会被长蛋白主导。

#### 8.2 Ranking/retrieval metrics

在每个 state 内计算，再按 protein/cluster 汇总：

```text
Spearman(Q, reward_lcb)
hit@1 / hit@5 / hit@10:
    top-k 是否至少包含一个 confirmed positive
precision@1 / precision@5 / precision@10:
    top-k 中 confirmed-positive 的比例
top-5% / top-10% hit rate
enrichment factor
NDCG (gain=max(reward_lcb, 0))
greedy regret:
    max_a reward_lcb(s,a) - reward_lcb(s,argmax Q)
greedy selected reward mean/median and bootstrap CI
strength/toughness delta and Pareto-positive fraction
PR-AUC for confirmed-positive retrieval
```

`positive@k` 容易产生歧义，日志中应同时明确记录 `hit@k` 和 `precision@k`。

bootstrap 必须以 protein 或 sequence cluster 为重采样单位，而不是把同一个 protein 的数千个
action rows 当成独立样本。test split 只在模型和超参数冻结后评估一次。

SFT best checkpoint 建议以 validation NDCG/top-k enrichment 为主、greedy regret 为辅，不以最低
train/validation regression loss 作为唯一标准，因为最终用途是动作选择。

### 9. 从 WT 扫描扩展到 RL 状态分布

只扫描 WT initial states 会让 visited flag 始终为0，也不会覆盖第2至24步的多突变序列。采用
迭代式数据聚合：

```text
Round 0:
    WT states full/sampled action scan -> SFT-v0

Round 1:
    SFT-v0 rollout to depths 1,2,4
    scan SFT top actions + ESM proposals + random diverse actions
    append unique states -> SFT-v1

Round 2:
    SFT-v1 rollout to depths 4,8,16
    repeat targeted scan -> SFT-v2

Evaluation:
    fixed greedy rollout to depth 24 on held-out proteins
```

中间 state 使用 sequence + pose checksum + visited mask 构成 state hash。对每个 state 可扫描：

```text
SFT top-32
ESM plausibility top-16
uniform/chemically-diverse random 16
```

这不是固定值，先用 Pilot 检查 coverage。模型建议必须与随机候选并存，否则数据聚合会只看到
当前模型已经相信的区域，无法纠正盲点。

### 10. A-D 四组策略的精确定义

#### A：Random initialization + current DDQN

```text
不加载 SFT checkpoint
不使用 demonstration quota
不使用 SFT proposal policy
```

这是所有实验的基线。若统一 sequence-cluster split 发生变化，A 必须重新训练。

#### B：SFT initialization + current DDQN

```text
online Q <- SFT Q head
target Q <- exact copy of online Q
new RL optimizer
no demonstration quota
no frozen SFT proposal mixture
```

B 与 A 只允许初始化不同。epsilon、PER、n-step、train frequency、reward、数据顺序和固定 greedy
validation 全部相同，以检验 SFT 初始化本身。

#### C：SFT initialization + high-confidence demonstration replay

先使用冻结 SFT policy 在 training proteins 上运行真实、完整的 environment trajectories。每一步
保存正确 observation、action、reward、next observation、done、visited mask 和 terminal LCB。
只有 episode-level terminal LCB 超过边界的完整轨迹进入 positive demonstrations；失败轨迹可以
作为 hard-negative demonstrations 单独标记。

训练 batch 中设置5%和10%两个子实验的 demonstration quota。按 episode/state 去重，保留正常
importance weighting。不能把同一成功 episode 的最后多个 n-step prefixes 当作多个独立 expert
episode。

#### D：SFT initialization + annealed SFT proposal/online-Q mixture

保留一个冻结的 SFT proposal network，与在线 Q network 分离。对非随机动作：

```text
with probability epsilon(t):
    uniform valid random action
otherwise:
    with probability eta(t):
        masked argmax frozen_SFT_Q(s,a)
    otherwise:
        masked argmax online_Q(s,a)
```

`eta(t)` 从起始值线性或 cosine 衰减到0。所有分支使用同一个 valid/revisit action mask。冻结 SFT
network 只运行轻量 Q head，可复用当前 ESM observation，不需要再次执行 ESM2 inference。

D 的重点是帮助早期 exploration，最终必须回到 online Q；若 eta 永不衰减，实验测量的是固定
SFT policy，而不是 DDQN 是否学会更好的策略。

第一阶段只做 A/B。只有 B 在固定验证集上优于 A，才运行 C/D。否则 C/D 会把一个无效 SFT
prior 更强地注入训练。

如 C 和 D 分别有效，之后可以增加 `E = C + D` 检查交互，但不应一开始就加入，否则四组结果
无法归因。

### 11. RL 接入和 warm-start

加载 SFT checkpoint 时必须检查：

```text
embedding_dim
state feature dimension (visited flag)
hidden_dims
20-AA ordering
per-residue action flattening convention
reward scale
dataset/reward contract compatibility
```

流程为：

1. 构造与 SFT 相同的 online Q network；
2. 加载 Q-head parameters；
3. 将 online parameters 完整复制给 target Q；
4. 新建 Adam optimizer；
5. environment_steps 和 optimization_steps 从0开始；
6. 使用短 learning-rate warmup，必要时初始 LR 为正常 DDQN LR 的0.1至0.25倍；
7. 可在前若干 optimizer steps 保留逐渐衰减的 SFT ranking auxiliary loss，作为后续 ablation，
   不放入最初的 B 组。

不要继承 SFT optimizer state，也不要只初始化 online network 而让 target network 保持随机。

### 12. A-D 的公平评估

所有组固定：

```text
same source/split manifests
same initial protein order
same PyRosetta seed schedule
same environment-step budget
same reward contract and LCB threshold
same DDQN/PER/n-step/target-sync parameters
same fixed greedy validation proteins and seeds
```

以 environment steps 而不是 wall-clock time 对齐。SFT 数据生成和预训练成本单独报告，不能隐藏，
但不计入在线 RL environment-step budget。

至少运行3个 RL seeds，并在相同 validation proteins 上做 paired bootstrap。主要比较：

```text
fixed-greedy terminal reward mean/median/LCB
positive episode fraction
strength/toughness deltas and Pareto-positive fraction
reward-vs-environment-steps AUC
首次达到 terminal reward >= 0 的 environment steps
depth 1/2/4/8/16/24 reward trajectory
action diversity and predictor OOD fraction
Q/return calibration, loss, TD error and grad norm（稳定性辅助指标）
```

SFT train loss 或 training replay reward 不能替代固定 greedy validation。

### 13. Go/No-Go 门槛

建议按以下顺序决策。

#### 数据生成 Pilot

继续扩展的最低条件：

```text
存在可重复的 positive-LCB actions；
positive density 不完全由少数蛋白贡献；
paired sign agreement 和 ICC 表明 action effect 大于结构噪声；
top candidates 没有集中在 predictor 明显 OOD 区域。
```

#### SFT held-out test

继续进入 RL 的最低条件：

```text
greedy hit@1/precision@1 的 cluster-bootstrap lower CI 高于 random baseline；
top-10% enrichment factor 显著大于1，2可作为有价值的初始目标；
median state-wise Spearman 为正；
greedy selected reward LCB 的均值/中位数不为负；
结果可在未见 sequence clusters 上复现。
```

#### RL A/B test

只有 B 相比 A 更早提高 fixed greedy reward、提高 reward AUC 或最终 CI，才进入 C/D。若 SFT
offline ranking 好但 B 无效，优先检查 TD catastrophic forgetting、Q scale、epsilon schedule 和
WT-to-multi-step distribution shift，而不是立即增加 demonstration 比例。

### 14. 关键风险与控制

```text
Predictor exploitation:
    保存 RF feature distance/tree dispersion，检查 top candidates，使用独立数据复核。

Data leakage:
    sequence-cluster group split；RL fixed validation 永不进入 SFT generation/tuning。

Positive-only collapse:
    listwise full action set + hard negatives + uncertainty-aware weights。

PyRosetta pseudo-replicates:
    验证 seed 确实改变结构轨迹；保存 replicate rows。

State distribution shift:
    iterative state aggregation，覆盖 visited mask 和多突变 states。

Wrong terminal labels:
    显式 finalization；不能把非终止 step 中的 terminal_reward=0 当真实零标签。

Wrong demonstration semantics:
    demonstration 使用完整环境轨迹，不能拼接独立 candidate scans。

Long proteins dominate:
    protein/state macro weighting and cluster-level bootstrap。

SFT Q scale mismatch:
    使用 RL-scale target，或保存/反变换 scaler；Huber + target clipping。

Structural partial observability:
    若 structure-feature probe 有效而 ESM2 probe 无效，向 observation 加入当前七特征、
    predicted strength/toughness 和 remaining horizon，而不是继续扩大 SFT。
```

### 15. 推荐执行顺序

```text
M0. 冻结 reward_contract、schema 和 source/split manifest。
M1. 在 CPU 服务器完成24蛋白 Pilot，全19替换 + null controls。
M2. 完成 replicate/LCB 分析，确认 positive density 和噪声边界。
M3. 在 GPU 服务器训练最小 SFT-v0，完成 cluster-held-out 排序评估。
M4. SFT-v0 通过 Go/No-Go 后，扩展至512至1000个蛋白。
M5. 使用 SFT rollout 聚合 depth 1/2/4/8 states，训练 SFT-v1。
M6. 实现并运行 A/B 单变量 DDQN 对照。
M7. B 有效后生成完整 high-confidence demonstrations，运行 C 的5%/10%。
M8. 实现冻结 SFT proposal mixture，运行 D。
M9. 若 C、D 均有效，再考虑 C+D 组合与更大数据规模。
```

### 16. 最终判断

将数据获取独立出来是正确选择，既能把 CPU 密集型 PyRosetta 与 GPU 密集型 ESM2/SFT 分离，
也能形成可复用、可版本化的数据资产。当前最需要坚持的原则是：

```text
数据生成与模型训练解耦；
ESM2 proposal 与 mechanical label 解耦；
监督 action ranking 与完整 demonstration trajectory 解耦；
SFT 初始化效果与 replay/proposal 策略效果逐组解耦。
```

按这一方案执行，A-D 的结果才能回答明确问题：A/B 判断模型初始化是否缓解奖励稀疏；C 判断
高置信完整轨迹是否改善 TD 学习；D 判断 SFT prior 是否改善早期探索。若三者仍不能提高固定
greedy mechanical reward，则主要瓶颈更可能是 reward predictor、结构状态可观测性或单点突变
的真实可达性，而不是 DDQN 训练技巧。

## 2026-09-16：CPU 数据生成、突变树深度与宽度澄清

### 17. 直接回答：不加载 ESM2 不等于纯随机搜索

CPU worker 不强制加载 ESM2，指的是**模型部署解耦**，不是把搜索策略降级为纯随机：

```text
ESM2/GPU stage:
    可选地为一批 frontier sequences 生成 proposal actions 或 embeddings/Q ranking

PyRosetta/CPU stage:
    读取已经生成的 candidate manifest
    独立构造结构并计算 point reward / repeated-relax LCB
```

如果某个 state 对全部 valid `19 * unvisited_positions` 动作做扫描，这一步是确定性的局部穷举，
不是随机扫描，也不需要 ESM2。真正需要控制的是：不能对穷举产生的每个 child 再继续全量穷举，
否则会形成指数级突变树。

如果不做全量扫描，则 candidate proposal 也不应只用 uniform random，可以混合：

```text
SFT/Q top-ranked candidates
ESM2 masked-token plausible candidates
按位置和氨基酸理化类别分层的 random candidates
模型高不确定性或 hard-negative candidates
```

CPU 机器可以完全不加载 ESM2：GPU 机器先读取 CPU 输出的 frontier sequences，批量生成下一轮
candidate manifest，再交还 CPU 扫描。两个阶段通过文件交换，仍然属于独立、可恢复的数据生成
pipeline。

### 18. 突变树中需要区分的三个“宽度”

对每个 state 必须区分：

```text
M = candidates_per_state
    在当前 state 真正执行 PyRosetta/reward evaluation 的候选 action 数。

C = children_per_parent
    一个 parent 在局部排序后最多提名多少 child。

B = global_beam_width
    对同一个 source protein，每个 depth 最终保留多少个 states。
```

用户提出的 `[1,3]` 如果指 `C` 且没有 global pruning，那么 `C=3, depth=8` 会产生：

```text
leaf states = 3^8 = 6,561
all states including root = (3^9 - 1) / (3 - 1) = 9,841
parents that require expansion through depth 8 = (3^8 - 1) / 2 = 3,280
```

对于平均长度约105 aa 的蛋白，每个 parent 全量扫描约 `105*19=1,995` 个 actions。一个初始
蛋白的 naïve ternary tree 就约需要：

```text
3,280 * 1,995 ~= 6.5 million PyRosetta candidate evaluations
```

因此 `[1^8, 3^8] * structure_count` 只计算了叶节点，没有计算内部节点，更没有乘上每个 parent
为了选择 child 所评估的候选 action 数；真实成本会更大。

### 19. 推荐结构：全局 beam，而不是完整分叉树

推荐每个 source protein 使用固定 global beam：

1. depth 0 只有 WT/current root；
2. 每个 beam state 评估 M 个候选 actions；
3. 所有 parent 的 candidate children 汇总；
4. 按 conservative reward、结构有效性和多样性统一排序；
5. 整个 depth 只保留 B 个 states，而不是每个 parent 都永久保留 B 个孩子。

复杂度近似为：

```text
root_candidate_count + (depth - 1) * B * M
```

例如：

```text
depth D = 8
global beam B = 3
root full scan M0 ~= 1,995
later candidates per beam state M = 64

total evaluations per protein
    ~= 1,995 + 7 * 3 * 64
    = 3,339

retained states including root
    = 1 + 8 * 3
    = 25
```

这与完整三叉树的约650万次 candidate evaluations 相比，降低了约三个数量级，同时仍然保留
多个 mutation paths。

### 20. 对 depth=8 的评估

depth 8 是一个合理的**中期目标**，因为它能让 visited mask 和多突变 sequence 真正发生变化，
也足以暴露 single-mutation SFT 在组合状态上的 distribution shift。但不建议第一轮就对全部
蛋白构建 depth-8 trees。

推荐分阶段：

```text
Pilot-0:
    depth=1，仅 WT full single-mutation scan
    目的：确认 positive density、噪声和 ESM2/SFT 可识别性

Pilot-1:
    depth=4, global beam=3, M=64
    目的：检查树搜索、state hashing、路径依赖和 epistasis

Scale-1:
    depth=8, global beam=3, M=64
    目的：生成 SFT-v1/v2 的多突变 states

Late-state coverage:
    depth=16/24 使用1至3条 rollout trajectories
    不再构建分支树
```

原因是 RL horizon=24，depth 8 仍只覆盖前中期。后期状态需要样本，但无需在 depth 16/24 继续
分叉；使用 SFT/online policy rollout 加少量随机扰动即可覆盖。

### 21. 对 width=[1,3] 的评估

如果 width 指 global beam width，则 `[1,3]` 合理，但建议解释为实验参数而不是随机整数：

```text
B=1:
    greedy path，成本最低；容易早期选错后永久失去其他路径；多样性不足。

B=2:
    一个 exploit path + 一个 diverse/uncertain path；适合大规模数据生成。

B=3:
    基本推荐值；可同时保留 reward-best、Pareto/diverse 和 exploratory path。
```

建议24蛋白 Pilot 使用 `B=3`，扩展到512至1000个蛋白后根据吞吐选择 `B=2` 或 `B=3`。
`B=1` 更适合作为 greedy rollout baseline，而不是主要 SFT 数据来源。

如果 width 指每个 parent 的 children 数，可以允许每个 parent 提名1至3个 children，但随后仍要
做 global pruning，使每层总 state 数不超过 B。不能让 parent-local width 在每层相乘。

### 22. 每个节点评估哪些候选 actions

#### 22.1 Round 0：还没有 SFT 模型

24蛋白 Pilot 的 WT root 建议全量扫描全部19种 substitutions，原因是：

1. 可以无偏估计真实 positive-action density；
2. 可以测量 ESM2 proposal 对 positive-LCB actions 的 recall；
3. 可以构成包含完整 positive/negative/near-zero surface 的 SFT-v0 数据；
4. 数据规模约29k actions，仍可控制。

扩展到更多蛋白时可用：

```text
M=64 candidates per WT state:
    24 ESM2-plausible proposals（GPU 预生成，可选）
    24 position/AA-class stratified candidates
    16 uniform random/diversity candidates
```

如果没有 GPU proposal，64个 candidates 全部使用分层覆盖和随机探索，而不是简单从 `L*19`
均匀抽样。位置可按 sequence length bins、二级结构/结构区域、当前氢键参与情况分层；mutant AA
按疏水、极性、带正电、带负电、芳香、特殊构象类别分层。

#### 22.2 Round 1 以后：已经存在 SFT-v0/v1

推荐每个中间 state 的 `M=64` 初始组合：

```text
24 SFT top-ranked actions
16 ESM2 plausible actions not already selected
16 stratified random/diverse actions
 8 uncertainty/hard-negative actions
```

比例需要根据 Pilot 的 positive recall 调整。所有集合先应用 no-op、visited-position 和结构合法性
mask，再去重。

如果 CPU 与 GPU 不在同一机器，采用交替批处理：

```text
CPU round k:
    完成 candidates -> 输出 retained frontier_sequences.parquet

GPU proposal round k+1:
    批量 ESM2 encode + frozen SFT ranking
    输出 candidate_manifest.parquet

CPU round k+1:
    读取 manifest 并执行 PyRosetta labels
```

没有必要让 CPU worker 实时 RPC 调用 ESM2；批次边界会让失败恢复和资源调度更简单。

### 23. 如何选择每层保留的 B 个 states

不能只保留 point reward 最大的 children，否则会重复选择噪声尖峰和高度相似路径。建议先排除：

```text
PyRosetta/feature extraction failures
明显 predictor OOD
违反 action/visited mask 的 candidates
严重结构能量异常或不满足质量约束的 candidates
```

然后按以下角色保留 `B=3`：

```text
beam slot 1: 最大 conservative absolute reward / reward LCB
beam slot 2: strength-toughness Pareto frontier 中与 slot 1 路径不同的 candidate
beam slot 3: 高不确定性、位置/氨基酸类别不同的 exploration candidate
```

初筛时所有候选通常只有 point reward，可用 point reward 减去经验 uncertainty penalty 排序；只对
top、near-threshold 和随机 control 做重复松弛，再使用 LCB 确认 retained child。若确认后 child
不再可靠，应回退到下一候选，而不是保留一个 false positive。

为避免三条 beam 很快收敛到近乎相同序列，可以加入最大突变集合重叠、最小 sequence Hamming
distance 或不同 mutation-position requirement。

### 24. 数据量与标签不会只来自 retained branches

每层只保留 B 个 child structures，不表示只把 B 个正样本写入 SFT dataset。对每个 parent 实际
评估的全部 M 个 candidates 都应写入 action table：

```text
M evaluated labels -> SFT positive/hard-negative/near-zero/negative examples
B retained child structures -> next-depth frontier
```

因此 beam search 同时解决两个问题：

1. 避免结构 tree 指数膨胀；
2. 保留 state 内完整的相对动作监督，不退化成 positive-only SFT。

未保留 candidate 通常无需保存 PDB，但必须保存 action、reward components、LCB/uncertainty、failure
reason 和结构特征变化。

### 25. 搜索空间是否仍然过大

完整 sequence mutation space 确实巨大，任何方法都无法穷举 `20^L`。本方案的目标不是覆盖全部
序列，而是构造足够多的、与 RL occupancy 接近的 state-action comparisons。

纯随机 depth-8 trajectory 的信息效率较低；纯 greedy trajectory 又会缺乏覆盖。推荐的 hybrid
beam 在每个 state 同时包含 model exploitation、ESM plausibility、结构/化学分层覆盖和随机探索，
再由真实 PyRosetta + predictor label 决定保留路径。这与“随机生成3个孩子”本质不同。

还应使用自适应 M：

```text
若 M=64 中已出现多个 confirmed positives：
    不继续扩大当前 state 的 candidate 数。

若没有 positive，且候选覆盖不足：
    扩展到 M=128 或执行该 state 的 full scan。

若 full/pilot scan 证明该蛋白几乎没有可靠 positive actions：
    停止扩树并记录为 negative protein，而不是浪费 depth budget。
```

### 26. 推荐的第一版参数

建议先冻结以下 Pilot 配置，而不是一开始在大量参数上网格搜索：

```text
proteins: 24，按 length/baseline mechanics/cluster 分层

Round 0:
    depth = 1
    root candidates = all valid 19 substitutions per position
    confirm repeats = 5 for screening, 10 for final high-confidence labels

Round 1:
    max depth = 4
    global beam width B = 3
    candidates per beam state M = 64
    globally retained states per depth = 3

Round 2（只有前两轮有效才执行）:
    max depth = 8
    global beam width B = 3
    candidates per beam state M = 64

Late-state check:
    1至3条 depth-16/24 rollout trajectories per protein
```

第一个核心对照不是 depth 8 的 B=1/2/3 网格，而是：

```text
full random/stratified proposals
vs
ESM2/SFT-guided hybrid proposals
```

在相同 PyRosetta evaluation budget 下比较：positive-LCB discovery rate、unique positive states、
top-k reward、路径多样性和 SFT held-out metrics。只有 guided proposal 在固定预算下显著提高正向
发现率，才证明引入 ESM2/SFT proposal 的额外 GPU 阶段值得。

### 27. 最终建议

用户提出的 depth 8 是合理目标，width 1至3 也处于合理范围，但 width 必须定义成**每层全局
beam width**，不能让每个 parent 的 children 数递归相乘。

最终推荐：

```text
不是随机完整突变树；
而是 WT 单点全扫描建立无偏基线，
随后用 B=3 的全局 beam 构建 depth-4 -> depth-8 多突变状态，
每个 state 评估 M=64 的 hybrid candidates，
全部 M labels 进入 SFT，只有 B structures 进入下一层。
```

CPU worker 不加载 ESM2 只是部署方式；proposal pipeline 仍然可以使用 ESM2。这样既能利用纯 CPU
服务器完成最昂贵的 PyRosetta 部分，又避免纯随机探索和指数级树增长。

## 2026-09-17：SFT 数据获取代码组织与 RL 边界设计

### 28. 组织原则

数据获取代码应与 RL 训练代码保持**单向依赖**：

```text
sft_data_generation
        |
        | 通过唯一的 rl_contract adapter 读取稳定能力
        v
existing environment / mutation / terminal reward implementation

training.py / asynchronous_training.py / agent / replay buffer
        X 不允许反向 import sft_data_generation
```

这样可以同时满足两个目标：

1. 不复制 PyRosetta mutation/repack 和七特征 mechanical predictor，避免 SFT label 与 RL reward
   悄悄产生实现漂移；
2. 数据获取代码不进入 RL 的训练调用链，数据生成失败、依赖变化或 CPU 调度不会影响现有 RL。

第一版建议放在当前仓库的独立顶层目录，便于直接进行 reward-contract integration test；接口稳定
后可把整个目录迁移成独立仓库。暂时不建议一开始复制现有 reward/environment 代码到另一个仓库，
因为两份实现很容易在参数和 bug fix 上分叉。

### 29. 推荐目录树

```text
Mechanical-protein-RL/
├── model/                         # 现有 RL，不改变原有职责
├── training.py                    # 现阶段不为数据生成修改
├── asynchronous_training.py       # 现阶段不为数据生成修改
│
├── sft_data_generation/           # 新增：完全独立的数据生成包
│   ├── README.md
│   ├── environment-cpu.yaml
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── round0_full_scan.yaml
│   │   ├── round1_depth4.yaml
│   │   └── round2_depth8.yaml
│   ├── schemas/
│   │   ├── source_manifest.schema.json
│   │   ├── state.schema.json
│   │   ├── action_task.schema.json
│   │   ├── replicate.schema.json
│   │   ├── action_label.schema.json
│   │   ├── frontier.schema.json
│   │   └── proposal.schema.json
│   ├── sft_data_generation/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── hashing.py
│   │   ├── records.py
│   │   ├── validation.py
│   │   │
│   │   ├── adapters/
│   │   │   └── rl_contract.py
│   │   │
│   │   ├── manifest/
│   │   │   ├── source.py
│   │   │   ├── sequence.py
│   │   │   ├── clustering.py
│   │   │   └── splitting.py
│   │   │
│   │   ├── candidates/
│   │   │   ├── full_scan.py
│   │   │   ├── stratified.py
│   │   │   ├── external_proposals.py
│   │   │   └── merge.py
│   │   │
│   │   ├── scan/
│   │   │   ├── tasks.py
│   │   │   ├── worker.py
│   │   │   ├── mutation.py
│   │   │   ├── scoring.py
│   │   │   └── shard_writer.py
│   │   │
│   │   ├── statistics/
│   │   │   ├── null_controls.py
│   │   │   ├── paired_lcb.py
│   │   │   ├── labels.py
│   │   │   └── summaries.py
│   │   │
│   │   ├── search/
│   │   │   ├── beam.py
│   │   │   ├── diversity.py
│   │   │   ├── frontier.py
│   │   │   └── stopping.py
│   │   │
│   │   └── pipeline/
│   │       ├── stages.py
│   │       ├── status.py
│   │       ├── resume.py
│   │       └── release.py
│   ├── scripts/
│   │   ├── prepare_source.sh
│   │   ├── run_round0.sh
│   │   ├── run_round1.sh
│   │   ├── run_round2.sh
│   │   ├── run_worker.sh
│   │   └── merge_release.sh
│   └── tests/
│       ├── test_action_contract.py
│       ├── test_beam_search.py
│       ├── test_clustering_split.py
│       ├── test_lcb.py
│       ├── test_resume.py
│       ├── test_schema.py
│       └── test_rl_contract_smoke.py
│
└── record/
    └── add_sft_train.md
```

数据生成代码不放进 `model/agent_module`、`environment_module` 或 `dataset_module`。它不是 online
RL 的一部分，也不应该增加现有 agent/environment 的职责。

### 30. 唯一允许接触 RL 的适配器

`sft_data_generation/adapters/rl_contract.py` 是数据项目唯一允许 import `model.*` 的文件。其他
数据生成模块只能调用适配器暴露的接口，不能直接访问 environment 私有成员。

建议适配器只暴露：

```python
class RLContractAdapter:
    def load_state(self, pdb_path, *, root_pdb_path=None): ...
    def clone_state(self, state): ...
    def valid_action_mask(self, state, visited_positions): ...
    def decode_action(self, action_index, sequence_length): ...
    def apply_action(self, state, action_index, *, seed): ...
    def evaluate_step_components(self, before, after): ...
    def evaluate_mechanical_properties(self, pose): ...
    def evaluate_terminal_improvement(self, root_pose, candidate_pose): ...
    def serialize_selected_state(self, state, path): ...
```

adapter 负责把现有环境参数、氨基酸顺序、local repack、step calculator 和 terminal predictor 映射
到稳定的数据生成接口。pipeline 本身只处理 state/action records，不知道 Gym、DDQN、replay
buffer 或 actor process 的存在。

严格禁止以下导入：

```text
training.py
asynchronous_training.py
model.agent_module
model.replay_buffer_module
model.asynchronous_module
model.logging_module
```

如果当前环境缺少某个真正需要的 public operation，优先在 adapter 内组合现有 public APIs；只有
无法保证一致性时，才把最小的纯 PyRosetta helper 提取为 RL 和 SFT 共用模块。不要为了数据生成
重构整个 environment。

### 31. 模块职责

#### 31.1 `manifest/`

只负责输入数据治理：

```text
读取当前 RL train index
解析单链 canonical sequence
PDB/sequence 唯一匹配
exact sequence dedup
sequence clustering
cluster-level SFT train/validation/test split
生成 source_manifest 和 split_manifest
```

它不导入 PyRosetta，也不计算 reward。这样可以先在轻量环境中独立检查数据泄漏和 split 统计。

#### 31.2 `candidates/`

只负责输出 candidate action indices 和来源：

```text
full_scan.py:
    root 的全部 valid L*19 actions

stratified.py:
    按位置、AA 理化类别和随机种子生成 model-free candidates

external_proposals.py:
    读取 GPU ESM2/SFT proposal manifest

merge.py:
    应用 action mask、去重、来源标记和 M budget
```

每个 action 记录 `proposal_sources`，同一个 action 可同时来自 SFT、ESM2 和 random，但只能执行
一次 PyRosetta。

#### 31.3 `scan/`

是唯一包含 PyRosetta CPU workers 的层：

```text
tasks.py:
    把 state + action + repeat seed 变成 deterministic task_id

worker.py:
    每个进程初始化一次 PyRosetta，按 state 分组处理 actions

mutation.py:
    通过 RLContractAdapter clone/apply action

scoring.py:
    输出 raw step components、mechanical predictions 和 terminal deltas

shard_writer.py:
    每个 worker 独立、原子写 Parquet shard
```

worker 不加载 ESM2、不训练模型、不选择 beam、不直接合并全局 CSV。单个 action 失败只产生带
failure reason 的记录，不应杀死整个 shard。

#### 31.4 `statistics/`

在 PyRosetta 扫描完成后离线运行：

```text
构建 no-op/null noise distribution
按 state/action 聚合 paired replicates
计算 mean, SD, sign agreement, LCB/UCB
分配 positive/hard-negative/near-zero/negative/failure labels
生成 protein/state/action-level summaries
```

统计层读取 immutable replicate rows，不调用 PyRosetta。更换 CI 方法时无需重新计算结构。

#### 31.5 `search/`

只处理 beam 状态：

```text
读取 aggregated action labels
应用 quality/OOD constraints
全局选择 B=3 frontier states
保证 sequence/path diversity
生成下一 depth 的 state manifest
检查 early stopping
```

它不生成 reward，只消费已有 labels。全局 beam 单元测试可以使用人工 reward table，不依赖
PyRosetta。

#### 31.6 `pipeline/`

负责 stage 编排、状态检查、resume 和 dataset release，不包含具体科学计算逻辑。每个 stage 都是
幂等的：相同 config hash 和输入 manifest 再执行时，应识别已有完整输出而不重复计算。

### 32. CLI 设计

不提供一个隐藏所有步骤的巨大脚本。每个阶段有显式命令，便于在不同机器执行和恢复：

```bash
# 1. 只需轻量 CPU：清洗、聚类、split
python -m sft_data_generation.cli prepare-manifest \
  --config sft_data_generation/configs/base.yaml

# 2. 生成 root full-scan tasks
python -m sft_data_generation.cli build-tasks \
  --config sft_data_generation/configs/round0_full_scan.yaml \
  --round 0 --depth 0

# 3. PyRosetta CPU 节点；多个 shard 可在多台机器执行
python -m sft_data_generation.cli scan-shard \
  --dataset-root /path/to/sft_dataset_v001 \
  --round 0 --depth 0 --shard-index 7 --num-shards 64

# 4. 合并、LCB 和 label
python -m sft_data_generation.cli aggregate \
  --dataset-root /path/to/sft_dataset_v001 \
  --round 0 --depth 0

# 5. 全局 beam 选择，输出 frontier
python -m sft_data_generation.cli select-frontier \
  --dataset-root /path/to/sft_dataset_v001 \
  --round 0 --depth 1 --beam-width 3

# 6. 数据验证和 release
python -m sft_data_generation.cli validate \
  --dataset-root /path/to/sft_dataset_v001

python -m sft_data_generation.cli release \
  --dataset-root /path/to/sft_dataset_v001 \
  --release-name sft_actions_v001
```

额外提供：

```bash
python -m sft_data_generation.cli status --dataset-root ...
python -m sft_data_generation.cli failed-tasks --dataset-root ...
python -m sft_data_generation.cli retry-manifest --dataset-root ...
```

这些命令输出 machine-readable JSON summary，shell 脚本只负责传参和调度，不复制业务逻辑。

### 33. CPU 与 GPU proposal 的文件接口

GPU proposal 不是 CPU scanner 的运行时依赖。两侧只交换 Parquet：

```text
CPU -> GPU: proposal_requests.parquet
    state_id
    sequence
    visited_mask
    valid_action_mask/checksum
    requested provider/model version

GPU -> CPU: proposal_responses.parquet
    state_id
    action_index
    esm_log_probability
    sft_score
    proposal_rank
    proposal_source
    ESM/SFT checkpoint checksums
```

CPU pipeline 验证 `state_id`、sequence/action-mask checksum 和 model checksum 后，再与 stratified
random candidates 合并。如果 response 缺失，可选择暂停该 stage 或按配置退化为 model-free
candidates，但不能静默改变 candidate composition。

这一接口允许：

1. CPU 服务器完全不安装 Torch/fair-esm；
2. GPU proposal 在当前训练服务器空闲时批量运行；
3. proposal 模型升级后保留旧 response，进行同预算对照；
4. 数据获取项目未来迁移到独立仓库时无需修改网络协议。

### 34. 输出数据目录

生成数据不写入 Git 仓库。推荐在大容量存储上使用内容和版本明确的目录：

```text
sft_dataset_v001/
├── metadata/
│   ├── dataset_version.json
│   ├── reward_contract.json
│   ├── config_resolved.yaml
│   ├── code_versions.json
│   └── checksums.sha256
├── manifests/
│   ├── source_manifest.parquet
│   ├── cluster_manifest.parquet
│   └── split_manifest.parquet
├── rounds/
│   ├── round_000/
│   │   ├── depth_000/
│   │   │   ├── states.parquet
│   │   │   ├── tasks/
│   │   │   ├── replicates/
│   │   │   ├── labels.parquet
│   │   │   ├── frontier.parquet
│   │   │   └── reports/
│   │   └── depth_001/
│   └── round_001/
├── proposals/
│   ├── requests/
│   └── responses/
├── structures/
│   ├── roots/
│   └── retained_states/
├── failures/
├── logs/
└── release/
    └── sft_actions_v001/
```

`release/` 只包含 train_sft 所需的稳定 schema；raw tasks、debug logs 和中间 PDB 不进入训练输入。

### 35. 核心 record/schema

建议使用明确的数据记录，而不是在不同阶段传递自由格式 dict：

```text
SourceRecord:
    protein_id, pdb_path, sequence, cluster_id, split

StateRecord:
    state_id, protein_id, parent_state_id, root_state_id,
    depth, sequence, visited_positions, pose_path/checksum

ActionTask:
    task_id, state_id, action_index, position, wt_aa, mutant_aa,
    repeat_seed, proposal_sources, config_hash

ReplicateRecord:
    task_id, success, failure_reason,
    before/after score components,
    strength/toughness before/after,
    terminal absolute/marginal reward,
    seven features before/after,
    timings and worker metadata

ActionLabel:
    state_id, action_index, repeat_count,
    mean, SD, LCB, UCB, sign agreement,
    label class, OOD/quality flags

FrontierRecord:
    state_id, source_action_id, depth, beam_rank,
    selection_score, diversity role, retained pose path
```

所有表包含 `schema_version`、`dataset_version`、`reward_contract_hash` 和 `config_hash`。不兼容版本
应显式失败，不能由 pandas 自动补 NaN 后继续训练。

### 36. 任务分片与恢复

task ID 应由不可变输入确定：

```text
task_id = hash(
    state_id,
    action_index,
    repeat_seed,
    reward_contract_hash
)
```

shard 分配使用 `hash(task_id) % num_shards`，因此增加 worker 不改变 task 内容。每个 shard：

1. 启动时读取已有结果 task IDs；
2. 只运行缺失任务；
3. 写入 `.tmp.parquet`；
4. 校验 row count、schema、finite fields 和 checksum；
5. 原子 rename 为完成文件；
6. 生成 `.complete.json`。

不要在 NAS 上让多个 worker 争抢同一个 SQLite write lock，也不要并发追加同一个 CSV。跨机器
并行使用静态 shards，单机内部再使用 process pool。

### 37. 配置分层

配置建议分为三个不可混合的部分：

```text
reward_contract:
    影响 label 语义；任何改变都必须创建新 dataset version。

search_config:
    depth, B, M, proposal composition, stopping and diversity；
    改变时创建新 round/experiment ID。

runtime_config:
    workers, shards, memory limits, output buffering and logging；
    不改变科学结果，可在 resume 时调整。
```

resolved config 在任务生成时冻结。worker 不能根据本机默认值修改科学参数。

### 38. 测试边界

#### 38.1 不需要 PyRosetta 的快速测试

```text
action index encode/decode round-trip
valid/no-op/visited mask
deterministic task IDs and shard assignment
schema round-trip
cluster split leakage
candidate merge/dedup and M budget
global beam B=3, not per-parent branching
paired LCB and label classes
resume and duplicate suppression
config/hash compatibility
```

#### 38.2 PyRosetta integration tests

使用一个很小的固定 PDB：

```text
相同 pose/action/seed：scanner 与 RL environment 得到相同 mutant sequence；
step reward components 一致；
terminal predictor 的七特征和 strength/toughness 一致；
candidate 相对 root 的 terminal improvement 一致；
不同 candidates 从同一个 parent clone 开始；
失败 action 生成失败记录而不是中断 shard。
```

这组 contract tests 是单向依赖能够长期安全存在的关键。每次修改 environment/reward 后只需运行
contract tests，不要求 RL test suite import 数据生成包。

#### 38.3 Pipeline smoke test

```text
1 protein
2 positions
3 candidate amino acids per position
2 repeat seeds
2 worker processes
```

验证 task -> scan -> aggregate -> labels -> beam -> resume -> release 全流程，并确认第二次执行不会
重复运行已完成 tasks。

### 39. 对现有 RL 文件的影响

实现数据获取阶段时，预期现有 RL 代码改动为：

```text
training.py                         0 changes
asynchronous_training.py            0 changes
model/agent_module                  0 changes
model/replay_buffer_module          0 changes
model/logging_module                0 changes
```

只允许在以下条件下做最小改动：adapter contract test 证明数据生成无法通过现有 public API 重现
RL mutation/reward。此时应提取一个无 RL 依赖的共享 helper，而不是让 scanner 调用 environment
private methods。

等数据 release 通过验证后，再作为下一阶段独立实现：

```text
train_sft.py + model/sft_module        监督训练
training.py --q-init-checkpoint        A/B 接入
demonstration/proposal flags            C/D 接入
```

不要在实现数据生成时提前修改 DDQN learner，从而保持问题定位和 Git 变更范围清晰。

### 40. 推荐实施顺序

```text
1. 创建 sft_data_generation scaffold、schema 和 config parser。
2. 实现 RLContractAdapter 和 scanner-vs-environment contract smoke test。
3. 实现 source manifest、sequence dedup、cluster split 和 leakage report。
4. 实现 root full-scan task builder、CPU worker、atomic shards 和 resume。
5. 实现 replicate aggregation、null controls、LCB 和 labels。
6. 实现 global beam B=3 与 retained-state PDB 输出。
7. 定义 proposal request/response schema，再接入 GPU ESM2/SFT proposal。
8. 完成1蛋白 pipeline smoke test。
9. 运行24蛋白 Round-0 Pilot；冻结首个 dataset release。
10. 数据质量通过后，才实现 train_sft.py 和 B/C/D 的 RL 接口。
```

### 41. 最终代码边界

第一版最稳妥的组织方式是：

```text
同一个 Git repository，独立 top-level package；
数据生成单向依赖一个 RL contract adapter；
现有 RL 代码零反向依赖、零运行时耦合；
CPU 与 GPU proposal 通过 Parquet 文件交换；
最终 SFT 只消费版本化 release，不读取 scanner 中间文件。
```

这样既符合“不干扰当前 RL 代码组织”的约束，也避免立即建立第二份会逐渐漂移的 PyRosetta/reward
实现。等 schema、adapter 和 contract tests 稳定后，`sft_data_generation/` 可以整体迁移到独立
仓库，当前 RL 只保留 dataset/checkpoint consumer。

## 2026-09-17：单机 SFT 数据生成与 Q-head 监督训练实现

### 42. 本次实现范围

本次按已确认的代码边界实现了两个彼此独立、可单独运行的阶段：

```text
Stage 1: sft_data_generation/
         cleaned PDB -> source/split manifest -> action tasks -> PyRosetta scan
         -> replicate LCB -> labels -> global beam -> dataset release

Stage 2: train_sft.py + model/sft_module/
         released states/actions -> frozen ESM2 -> supervised per-residue Q head
```

没有修改以下在线强化学习代码：

```text
training.py
asynchronous_training.py
model/agent_module/ddqn_agent.py
model/replay_buffer_module/replay_buffer.py
```

因此当前 DDQN 行为和已运行基线不受影响。SFT checkpoint 接入 DDQN、A/B/C/D 对照仍作为下一
阶段单独实现，避免数据生成、监督训练和 RL warm-start 同时变化后无法定位问题。

### 43. 当前 RL 使用的 cleaned PDB 数据

当前 `train_version_terminal_async.sh` 实际指定的数据为：

```text
PDB directory:
/mnt/nas/jianquanzhao/data/mprl/pdbs/cath

RL train index:
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt

fixed RL validation index:
/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt
```

现场核对结果：

```text
PDB files in directory: 8,557
RL train index:          7,701 unique entries
RL validation index:       856 unique entries
train/validation overlap:    0
```

SFT source 只读取这7,701个 RL training entries；856个 fixed RL validation entries 被显式保留，
不参与 SFT 数据生成、监督训练和超参数选择。这样以后 A/B 对照仍可以使用原固定验证集。

目录虽然是 RL 使用的预清洗 PDB 集，但 environment 加载时仍可能删除单个缺少 N/CA/C/O 的残基。
真实 contract smoke 中，文本 PDB 有33个表面合法残基，而 PyRosetta Pose 为32个残基。因此 source
manifest 不再仅依赖 PDB 文本顺序，而是通过与 RL 相同的 environment/PyRosetta loader 获得最终
sequence 和 action length。这避免离线 `position * 20 + aa` 与在线 Pose position 发生偏移。

### 44. 存储目录

已创建用户指定的单机目录：

```text
/mnt/nas/jianquanzhao/data/mprl/SFT/
├── data/
│   ├── source/
│   └── generated/
├── model/
├── log/
└── tensorboard/
```

正式数据版本位于 `data/generated/<dataset_version>/`；SFT 模型、训练日志和 TensorBoard 分别按
`<run_version>_<YYYYmmdd_HHMMSS>` 创建子目录，不依赖另一台机器同步。

### 45. 数据生成实现

新增顶层独立包 `sft_data_generation/`，主要能力如下。

#### 45.1 Source manifest 与 split

`manifest.py` 实现：

1. 读取当前 RL train index；
2. 通过 environment loader 获取实际 PyRosetta sequence 和 chain count；
3. 将 WT PDB copy/symlink 到 `data/source`；
4. exact-sequence dedup，只保留 canonical structure 进入主扫描；
5. 可选 MMseqs2 sequence-similarity clustering；
6. 按完整 cluster 做80/10/10 SFT train/validation/test split；
7. 小样本下只要 cluster 数足够，保证三个非零 split 不为空；
8. 检查同一 cluster 不跨 split。

生产配置 `configs/base.yaml` 使用 MMseqs2：

```text
minimum sequence identity = 0.30
coverage = 0.80
split seed = 20260917
```

`configs/smoke.yaml` 使用 exact clustering，仅用于当前服务器未安装 MMseqs2 时做 dependency-light
smoke test，不能替代最终的 sequence-similarity split。

#### 45.2 Reward contract 与 task

`contract.py` 冻结并哈希：

```text
20-AA ordering = ACDEFGHIKLMNPQRSTVWY
local repack radius = 8 A
perform repack = true
perform minimize = false
step reward scale = 0.025
terminal reward scale = 8.0
random-forest artifact path and SHA256
code Git commit
PDB cleaning parameters
```

`tasks.py` 对 root state 枚举所有非 no-op、非 visited 的 `L * 19` actions，并用
`state_id + action_index + repeat_seed + reward_contract_hash` 生成确定性 task ID。重复启动、改变
worker 数或 shard 数不会改变任务语义。

#### 45.3 PyRosetta scanner

`adapters/rl_contract.py` 是唯一 import `model.*` 的离线文件。它直接复用：

```text
MechanicalProteinEnv mutation/action mapping
local repack implementation
StepRewardCalculator
MechanicalImprovementTerminalRewardCalculator
params/hbond_random_forest.joblib
```

每个 task 从同一 parent PDB 重新 reset，`max_steps=1` 保证 terminal reward 被真实计算，而不是把
非终止 step 的0误当标签。每条 replicate 同时保存：

```text
step reward
marginal terminal reward relative to parent
absolute terminal reward relative to root
strength/toughness marginal and absolute delta
before/after physical predictions
seven features before/after
candidate sequence, failure reason and runtime
raw reward and RL-scaled reward
```

PyRosetta seed 通过 `numeric.random.rg().set_seed()` 设置，并记录是否成功应用。task 失败只产生
failure row，不中止整个 shard。每个 shard 使用独立 JSONL progress file，重启时跳过已有 task ID，
完成后输出 CSV 和 `.complete.json`。

#### 45.4 LCB、标签和 beam

`statistics.py` 保存 replicate rows，并按 state/action 计算 bootstrap mean CI、SD、sign agreement
和 confidence weight。标签为：

```text
confirmed_positive
hard_negative
near_zero_or_uncertain
confirmed_negative
structural_failure
```

阈值、置信度、bootstrap 次数和目标列均由 YAML 控制。`search.py` 执行每个 root 的全局
`beam=3`，不是每个 parent 各保留3个 children。下一深度 state 必须拥有已 materialize 的候选 PDB，
避免只有 sequence 没有结构仍继续搜索。

#### 45.5 Round 1-K 候选

`proposals.py` 接受同机产生的 ESM2/SFT proposal tables，应用 action mask、来源合并、action 去重和
固定 `M=64` budget；默认至少包含16个确定性随机候选以保持探索覆盖。如果未提供模型 proposal，
会明确生成 model-free random candidates，不会静默标记为 ESM2/SFT 数据。

#### 45.6 Release

`release.py` 在输出训练集前检查 schema、未知 state 和 cluster leakage，稳定输出：

```text
states.csv
actions.csv
dataset_version.json
```

CSV/JSONL 是不依赖 PyArrow 的默认交换格式；I/O 同时支持 Parquet，安装 `pyarrow` 后可直接使用
`.parquet` 后缀。

### 46. SFT Q-head 训练实现

新增：

```text
train_sft.py
model/sft_module/dataset.py
model/sft_module/losses.py
model/sft_module/metrics.py
model/sft_module/trainer.py
```

训练单位为 state，不是独立 action row。variable-length batch 对序列和 `L * 20` 标签做 padding，
使用 `label_available_mask`，未扫描 action 不会被当成 reward=0。

模型与 DDQN 保持同构：

```text
frozen ESM2 per-residue embedding (default 1280)
+ visited-position feature
-> shared residue MLP (default 256, 256)
-> 20 action scores per residue
```

联合 loss 为：

```text
L = 1.0 * listwise KL
  + 0.5 * positive-vs-negative margin ranking
  + 0.5 * confidence-weighted Huber regression
```

训练记录 MAE、RMSE、state-macro Spearman、NDCG、positive hit@1/5/10、greedy regret、greedy
selected reward 和 Pareto-positive fraction。best checkpoint 以 validation macro NDCG 选择，包含：

```text
Q-head state_dict
architecture and 20-AA ordering
visited-mask configuration
target column and loss configuration
dataset path
best validation metrics
SFT optimizer/scheduler state for resume
```

后续 DDQN 初始化只应读取 Q-head 和 compatibility metadata，不继承 SFT optimizer momentum。

### 47. 环境包与迁移方法

提供三份环境文件：

```text
sft_data_generation/environment-cpu.yaml    PyRosetta 数据生成
sft_data_generation/environment-train.yaml  ESM2 + Q-head GPU 训练
sft_data_generation/environment-all.yaml    单机完整流程
```

另一台 CPU 服务器：

```bash
conda env create -f sft_data_generation/environment-cpu.yaml
conda activate mprl-sft-data
```

GPU SFT 服务器：

```bash
conda env create -f sft_data_generation/environment-train.yaml
conda activate mprl-sft-train
```

本阶段单机完整环境：

```bash
conda env create -f sft_data_generation/environment-all.yaml
conda activate mprl-sft
```

CPU 标签生成的必要包为：

```text
Python 3.10
PyRosetta
NumPy, pandas, SciPy
scikit-learn 1.6.1
joblib, PyYAML, tqdm
MMseqs2
PyArrow（仅 Parquet 必需）
```

`params/hbond_random_forest.joblib` 是由 scikit-learn 1.6.1 序列化的。当前 `mprl-vgpt` 使用
1.7.2，真实 smoke 能完成但会产生 `InconsistentVersionWarning`；正式生成标签必须使用提供的
1.6.1 pin，避免版本差异导致树模型行为风险。

GPU 训练额外需要：

```text
PyTorch >= 2.0
fair-esm == 2.0.0
TensorBoard
Matplotlib
```

GPU SFT 不需要 PyRosetta；CPU generator 不需要 Torch/ESM2。

### 48. 使用方法

准备 source、聚类和 split：

```bash
bash sft_data_generation/scripts/prepare_source.sh
```

单机 Round-0 分片扫描：

```bash
SFT_WORKERS=8 SFT_SHARDS=64 \
bash sft_data_generation/scripts/run_round0.sh
```

发布数据后训练 Q head：

```bash
SFT_RELEASE_DIR=/path/to/sft_actions_v001 \
SFT_ESM_MODEL_DIR=/mnt/data1/home/jianquanzhao/data/models/esm \
bash sft_data_generation/scripts/train_sft.sh
```

详细命令、schema 和目录说明见 `sft_data_generation/README.md`。

### 49. 测试结果

新增纯 CPU 测试：

```text
tests/test_sft_data_generation.py
tests/test_sft_module.py

7 passed
```

覆盖：

```text
PDB sequence parsing
action encode/decode round trip
deterministic task IDs and shard assignment
bootstrap LCB and action classes
cluster leakage rejection
proposal dedup/budget/random coverage
global beam semantics
variable-length collate and masks
joint SFT loss backward
Q-head train/evaluate loop
```

现有相关回归测试：

```text
tests/test_ddqn_agent.py
tests/test_environment.py
tests/test_terminal_reward_calculator.py

51 passed
```

真实 PyRosetta contract smoke：

```text
PDB: 1a0iA02_2.pdb
environment sequence before: ACTINPYDGWACQISYMEETPDGSLRPSFVMF
action: position 1, A -> C
sequence after: CCTINPYDGWACQISYMEETPDGSLRPSFVMF
mutation/repack/step reward/random-forest terminal delta: completed
offline sequence/action assertions: passed
```

4-PDB pipeline smoke 已在指定 NAS 目录完成：

```text
dataset root:
/mnt/nas/jianquanzhao/data/mprl/SFT/data/generated/sft_smoke

source entries: 4
clusters: 4
split: train=2, validation=1, test=1
Round-0 tasks: 21,736 (2 repeat seeds)
```

三个 shell 脚本均通过 `bash -n`。尚未启动7,701个 proteins 的正式 MMseqs2 聚类和完整
PyRosetta full scan；当前服务器未安装 MMseqs2，且正式扫描属于长时间 CPU 作业。安装
`environment-cpu.yaml` 后再运行 production config，避免把 exact smoke split 当作正式实验数据。

### 50. 当前结论

本次实现满足“单机运行、数据生成与 RL 解耦、复用完全相同 reward contract、输出版本化训练数据、
训练同构 Q head”的目标。最重要的工程验证不是简单 import 成功，而是发现并修复了 PDB 文本残基
与 PyRosetta 实际 Pose 残基不一致造成的 action-position drift。下一步应先在24个 proteins 上运行
Round-0 pilot，确认 positive-LCB density、repack repeatability 和 predictor OOD，再决定是否扩展到
全部7,701个 training proteins；不建议立即提交整批 full scan。

## 2026-09-20：SFT 与 DDQN Q-network 架构对齐及启动脚本

### 51. 当前 RL Q-network 的实际架构

当前异步 RL 启动脚本使用以下网络参数：

```text
ESM2 model                    esm2_t33_650M_UR50D
ESM2 per-residue embedding    1280
visited mask                  1 feature/residue
Q-head input                  1281 = 1280 + 1
hidden dimensions             256, 256
amino-acid actions            20/residue
Q-head layers                 1281 -> 256 -> ReLU -> 256 -> ReLU -> 20
```

对于 batch size 为 `B`、序列长度为 `L` 的蛋白，Q-network 输入为
`[B, L, 1281]`，输出为 `[B, L * 20]`。该网络是共享参数的 per-residue head：同一个
MLP 独立映射每个残基的特征，因此 `L` 只改变输出动作数量，不改变参数形状。

可训练 Q head 的参数形状与参数量为：

```text
residue_head.0.weight    [256, 1281]
residue_head.0.bias      [256]
residue_head.2.weight    [256, 256]
residue_head.2.bias      [256]
residue_head.4.weight    [20, 256]
residue_head.4.bias      [20]
total trainable parameters: 399,124
```

ESM2 encoder 当前保持冻结，不包含在上述399,124个 Q-head 参数中。online Q network 与
target Q network 均由同一个 `QNetwork` 类构建，架构完全相同，仅参数更新方式不同。

### 52. SFT/DDQN 兼容性保证

SFT 训练不再复制一份近似网络，而是直接实例化 RL 使用的
`model.agent_module.ddqn_agent.QNetwork`。新增
`model/sft_module/compatibility.py`，在训练开始前执行以下检查：

1. SFT 和 DDQN 均进入 per-residue 模式；
2. 完整比较两者 `state_dict` 的名称和参数形状；
3. 使用 `strict=True` 将 SFT 参数加载到一个不同序列长度的 DDQN network；
4. 对变长输入执行 forward，并验证输出为 `L * 20`；
5. 将完整的 `q_network_contract` 写入 SFT checkpoint，便于 RL 加载时审计。

也可以不读取数据、不加载 ESM2，单独检查架构：

```bash
/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python train_sft.py \
  --data-dir /tmp/not-used \
  --embedding-dim 1280 \
  --hidden-dims 256,256 \
  --include-visited-mask \
  --check-architecture-only
```

本次检查结果为 `1281 -> 256 -> 256 -> 20`、399,124个参数；使用长度37的
probe protein 时，batch size 2 的输出为 `[2, 740]`，严格 checkpoint 加载通过。

### 53. 一键启动脚本

根目录新增 `train_sft.sh`，默认使用以下目录：

```text
dataset       /mnt/nas/jianquanzhao/data/mprl/SFT/data/generated/sft_v001/release/sft_actions_v001
model         /mnt/nas/jianquanzhao/data/mprl/SFT/model/<run_id>
log           /mnt/nas/jianquanzhao/data/mprl/SFT/log/<run_id>
tensorboard   /mnt/nas/jianquanzhao/data/mprl/SFT/tensorboard/<run_id>
ESM2          /mnt/data1/home/jianquanzhao/data/models/esm
Python        /home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python
```

`model`、`log` 和 `tensorboard` 使用完全相同的 `<run_id>`，默认格式为
`sft_qhead_v001_YYYYmmdd_HHMMSS`。启动前会检查 release 文件、ESM2 checkpoint、
PyTorch/CUDA、真实 GPU tensor allocation，以及 SFT/DDQN 架构契约。默认在后台运行并写入
`train.pid` 和 `train_stdout.log`。

正式 release 就绪后，可使用：

```bash
SFT_DATA_DIR=/mnt/nas/jianquanzhao/data/mprl/SFT/data/generated/<version>/release/<release> \
SFT_VISIBLE_GPUS=0 \
bash train_sft.sh
```

前台运行并同步查看 stdout：

```bash
SFT_BACKGROUND=0 \
SFT_DATA_DIR=/mnt/nas/jianquanzhao/data/mprl/SFT/data/generated/<version>/release/<release> \
bash train_sft.sh
```

常用覆盖项包括 `SFT_RUN_VERSION`、`SFT_RUN_ID`、`SFT_EPOCHS`、
`SFT_STATE_BATCH_SIZE`、`SFT_LEARNING_RATE`、`SFT_WEIGHT_DECAY`、
`SFT_WARMUP_RATIO`、`SFT_VISIBLE_GPUS` 和 `SFT_DEVICE`。脚本末尾还可追加
`train_sft.py` 的其他参数。旧入口 `sft_data_generation/scripts/train_sft.sh` 已改为转发到
根目录脚本，防止两套训练配置发生漂移。

### 54. 测试和当前边界

```text
bash syntax checks: passed
Python compile checks: passed
architecture-only compatibility check: passed
SFT/DDQN/environment/terminal-reward tests: 59 passed
git diff whitespace check: passed
```

当前默认 release 目录尚未生成 `states.csv` 和 `actions.csv`，因此没有启动正式 SFT 长任务。
这是预期的 fail-fast 行为：脚本会明确提示设置 `SFT_DATA_DIR`，避免在错误或未发布的数据上训练。

## 2026-10-12：SFT 训练代码与 `sft_v001` 数据就绪度审计

### 55. 审计结论

本次结论需要区分“代码实现完成”“数据生成完成”和“可以直接启动正式训练”三个层次：

| 检查对象 | 状态 | 结论 |
| --- | --- | --- |
| 独立 SFT Q-head 训练模块 | 已完成 | 网络、损失、指标、checkpoint、TensorBoard 和恢复训练代码均已实现 |
| SFT/DDQN 架构兼容 | 已通过 | SFT 直接复用 RL 的 `QNetwork`，严格参数加载和变长 forward 均通过 |
| Round 0 训练集单点突变扫描 | 已完成 | 6145 个训练蛋白、11,873,290 个动作全部执行成功，无失败任务 |
| validation/test 动作标签 | 未生成 | 当前所有动作标签均属于 train split，不能进行独立验证和测试 |
| SFT release | 未生成 | 缺少训练入口要求的 `states.csv`、`actions.csv` 和 `dataset_version.json` |
| 重复松弛与统计 LCB | 未完成 | 每个动作只有一个 seed，当前 `reward_lcb` 实际等于单次 `reward_mean` |
| SFT checkpoint 自动接入 RL | 未完成 | 当前可训练兼容 checkpoint，但 RL 的 A/B/C/D 实验入口仍需单独接入 |

因此，**SFT 训练代码主体已经完成且网络架构可行，但当前 `sft_v001` 还不能直接用于正式 SFT 训练**。
它已经完成了规模很大的 Round 0 训练集标签生成，但还处于“聚合后的中间数据”阶段，而不是可发布、可验证的训练数据集。

### 56. 当前 SFT 代码完成了什么

当前 SFT 代码包含以下完整训练链路：

1. `train_sft.py`：参数解析、随机种子、设备选择、数据加载、训练、验证、checkpoint 和 TensorBoard 入口；
2. `model/sft_module/dataset.py`：按照 state 组织候选动作，并支持不同蛋白长度的 batch collate；
3. `model/sft_module/losses.py`：组合 listwise KL、pairwise ranking 和 Huber regression；
4. `model/sft_module/metrics.py`：计算 state-macro Spearman、NDCG、positive@K、top hit rate、greedy regret、MAE/RMSE 和 Pareto-positive 等指标；
5. `model/sft_module/trainer.py`：冻结 ESM2，只训练 Q head，并实现 AdamW、warmup、cosine learning-rate schedule、梯度裁剪和 early stopping；
6. `model/sft_module/compatibility.py`：检查 SFT 与 DDQN 的参数名称、形状、输入契约和严格 checkpoint 加载；
7. `train_sft.sh`：统一数据、模型、日志和 TensorBoard 路径，并执行启动前检查；
8. `sft_data_generation/`：与 RL 主训练代码分离的 CPU 数据生成、分片执行、恢复、聚合和 release 工具。

SFT 与 RL 当前使用相同的 per-residue Q head：

```text
ESM2 residue embedding       1280
visited-position mask           1
Q-head input dimension       1281
hidden dimensions         256, 256
amino-acid action dimension    20
trainable parameters       399,124
```

其映射为 `1281 -> 256 -> 256 -> 20`。对长度为 `L` 的蛋白，网络输入为
`[B, L, 1281]`，输出展平为 `[B, L * 20]`。因此蛋白长度改变不会改变 Q head 参数形状。

本次重新执行了以下验证：

```text
shell syntax checks: passed
SFT/DDQN strict architecture compatibility: passed
variable-length forward probe: passed
focused pytest suite: 41 passed in 10.24 s
```

这里的“通过”证明网络和训练模块的程序契约成立，不代表当前 11,873,290 条标签已经满足正式训练的数据条件。

### 57. 当前启动脚本中的明确问题

根目录 `train_sft.sh` 当前连续设置了两次 `PYTHON_BIN`：

```bash
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda3_v24/envs/mprl-sft/bin/python}"
```

第二行覆盖第一行。实机检查结果为：

```text
/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python      exists, Python 3.10.20
/home/jianquanzhao/anaconda3_v24/envs/mprl-sft/bin/python   missing
```

所以即使 release 已经生成，当前不设置 `SFT_PYTHON` 时也会在启动前失败。正式启动前应删除第二次覆盖，或显式设置：

```bash
SFT_PYTHON=/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python bash train_sft.sh
```

### 58. `sft_v001` 文件夹的实际组成

当前目录实际结构可概括为：

```text
sft_v001/
├── manifests/
│   ├── source_manifest.csv
│   ├── split_manifest.csv
│   └── mmseqs/
│       ├── sequences.fasta
│       ├── mmseqs_clusters_all_seqs.fasta
│       ├── mmseqs_clusters_rep_seq.fasta
│       ├── mmseqs_clusters_cluster.tsv
│       └── mmseqs_tmp/
├── metadata/
│   ├── config_resolved.yaml
│   └── reward_contract.json
└── rounds/
    └── round_000/
        └── depth_000/
            ├── tasks.csv
            ├── replicates/
            │   ├── shard_*.jsonl
            │   ├── shard_*.csv
            │   └── shard_*.complete.json
            ├── replicates.csv
            └── labels.csv
```

当前没有 `release/`、`frontier.csv` 或 `depth_001` 及更深层目录。

### 59. manifests：蛋白来源、去重和数据划分

`source_manifest.csv` 是 WT 数据清单。每一行对应一个通过清洗的原始蛋白，记录：

- protein/source ID；
- 原始和物化后的 PDB 路径；
- 与 PyRosetta pose 对齐后的氨基酸序列和长度；
- 文件与序列 hash；
- 是否为重复项及 canonical entry 信息；
- 数据是否有效以及排除原因。

`split_manifest.csv` 在 source manifest 基础上加入：

- MMseqs sequence cluster；
- `train`、`validation` 或 `test` split；
- `state_id`、`root_state_id`、`parent_state_id`、depth 和 visited-position 等搜索状态字段。

数据不是逐条随机划分，而是先按序列相似度聚类，再按 cluster 划分，从而避免同源序列跨 train/validation/test 泄漏。

本次读取到的划分统计为：

| 项目 | 数量 |
| --- | ---: |
| source entries | 7701 |
| 纳入 SFT split | 7694 |
| 排除/保留给 RL validation | 7 |
| train proteins | 6145 |
| validation proteins | 770 |
| test proteins | 779 |
| sequence clusters | 5538 |
| train/validation/test clusters | 4430 / 554 / 554 |
| cluster 跨 split 泄漏 | 0 |

蛋白长度范围为 26--584 aa，中位数为 89 aa。聚类参数来自
`config_resolved.yaml`：sequence identity 0.3、coverage 0.8、划分比例 0.8/0.1/0.1、seed 20260917。

`manifests/mmseqs/` 保存聚类输入、代表序列、cluster 映射和临时数据库。它主要用于复现数据划分，不是 `train_sft.py` 直接读取的训练输入。

### 60. metadata：实验配置和 reward 契约

`config_resolved.yaml` 是本次数据生成任务的完整解析后配置，记录目录、beam/depth、候选生成、重复 seed、PyRosetta 和 release 参数。它用于确保离线生成可以复现。

`reward_contract.json` 固化了标签生成规则，包括：

- amino-acid action 顺序；
- random forest 文件路径和 SHA-256；
- repack/minimize 设置及局部半径；
- strength/toughness 到 scalar reward 的组合方式；
- step/terminal scale；
- git commit 和整个 contract 的 hash。

当前 reward predictor 为 `params/hbond_random_forest.joblib`；配置启用 repack、未启用 minimize、local radius 为 8、step scale 为 0.025、terminal scale 为 8。该契约应随最终 release 一起保存，防止 SFT label 与 RL reward 定义悄然漂移。

### 61. rounds：动作规划、实际执行和标签聚合

`round_000/depth_000/tasks.csv` 是待执行任务表。每行表示一个：

```text
state + residue position + mutant amino acid + repeat seed
```

当前 Round 0 明确只对 `train` split 的 WT state 做 depth=0 全单点突变扫描。6145 个训练蛋白总长度为 624,910 aa；每个位置排除 WT 后有 19 种候选，因此理论动作数为：

```text
624,910 * 19 = 11,873,290
```

`tasks.csv` 正好包含 11,873,290 个任务。

`replicates/shard_*.jsonl` 是每个 CPU shard 的追加式进度日志，用于中断恢复；
`shard_*.csv` 是标准化结果；`shard_*.complete.json` 记录 shard 的任务数、成功数、失败数和耗时。当前 64 个 shard 均存在完成标记，assigned、executed 和 succeeded 均合计为 11,873,290，failed 为 0。

`replicates.csv` 是合并并去重后的原始执行结果。它保留 mutation 前后结构特征、step reward 分量、strength/toughness 变化、终局 reward 以及失败信息，约 39.19 GB。

`labels.csv` 将同一 state/action 的重复实验聚合为一行，预期提供：

- `reward_mean`、`reward_std`、`reward_lcb`、`reward_ucb`；
- positive/negative/uncertain 标签；
- confidence weight；
- strength/toughness delta；
- Pareto-positive 标记。

当前该文件约 10.24 GB，共 11,873,290 行。

### 62. 全量标签审计结果

对整个 `labels.csv` 进行了流式扫描，而不是只查看头部样本：

| 指标 | 结果 |
| --- | ---: |
| finite reward rows | 11,873,290 |
| covered states/proteins | 6145 / 6145 |
| confirmed positive | 5,297,874（44.6201%） |
| confirmed negative | 6,277,043（52.8669%） |
| near zero or uncertain | 298,373（2.5129%） |
| scalar reward mean | -0.049894 |
| Pareto-positive | 24.0748% |
| `reward_std == 0` | 100% |
| `reward_lcb == reward_mean` | 100% |

从 Round 0 单点突变标签看，正 reward 动作占 44.62%，因此这里不存在“单点正动作在离线标签中极度稀少”的现象。RL 中观察到的 reward 稀疏或持续为负，更可能来自：

1. 连续多步突变后状态分布与 WT 单点扫描不同；
2. PyRosetta 结构更新和 predictor 噪声累积；
3. strength/toughness 的冲突与 toughness 整体下降；
4. exploration 访问到的动作分布与全扫描标签分布不同；
5. Q head 尚未学习每个 state 内部的正确动作排序。

当前 scalar reward 均值为负，主要与 toughness delta 的均值更负有关：strength delta 均值约为 -0.001455，而 toughness delta 均值约为 -0.011018。

另一个关键问题是近零标签。全量数据中：

```text
|reward| <= 1e-12 : 14.4670%
|reward| <= 1e-3  : 15.0859%
|reward| <= 1e-2  : 19.3854%
|reward| <= 5e-2  : 32.4461%
|reward| <= 1e-1  : 45.5409%
```

但是只有 2.51% 被标记为 near-zero/uncertain。这意味着约 11.95% 的动作几乎只有浮点数符号差异，却被归入 confirmed positive 或 confirmed negative。若直接用于 ranking/classification loss，模型会被要求学习超出 reward predictor 分辨率的差异。

### 63. 为什么当前 LCB 还不是真正的噪声下置信界

`config_resolved.yaml` 中当前只有：

```yaml
repeat_seeds: [11]
```

所以每个 action 的 requested repeat 和 successful repeat 都是 1。由此导致：

- 无法从重复 repack/relax 估计方差；
- `reward_std` 全部为 0；
- `reward_lcb` 全部等于 `reward_mean`；
- “LCB > 0”与“单次 reward > 0”完全相同；
- 当前 confidence weight 不能表达 PyRosetta/reward predictor 的重复性。

因此 `labels.csv` 虽然具备 LCB 字段格式，但当前不应将其解释成“超过结构松弛噪声后的高置信正样本”。这也是正式 SFT 前最重要的统计边界。

### 64. 为什么当前数据不能直接交给 `train_sft.py`

训练入口读取的是一个稳定 release，而不是直接读取 `rounds/` 中的生成中间文件。期望目录为：

```text
release/<release_name>/
├── states.csv
├── actions.csv
└── dataset_version.json
```

三者含义为：

- `states.csv`：state 级信息，包括 split、sequence、结构/state lineage 和 visited positions；
- `actions.csv`：用于监督的 state/action 标签，只保留训练需要的列；
- `dataset_version.json`：源数据、reward contract、schema、行数和文件 hash，防止误用不同版本数据；

当前 `sft_v001` 没有 `release/`。此外，现有动作标签 100% 属于 train split；validation 理论上还需评估 1,598,831 个单点动作，test 理论上还需 1,518,119 个。若仅把当前 train labels 打包为 release，validation dataset 会因为找不到可用动作标签而失败，early stopping 和所有 validation ranking 指标也无法计算。

还有一个工程风险：当前 `SFTActionDataset` 分别为 train 和 validation 加载完整 CSV，并在内存中按 state 组织。对 10 GB 级 `actions.csv`，即使生成了 release，也会造成很长的启动时间和很高的内存占用。因此正式训练前还需要 split-specific 文件、Parquet/Arrow，或分块索引式读取，避免同一巨型 CSV 被重复完整加载。

### 65. 可行性判断和推荐顺序

#### 65.1 可以确认可行的部分

1. 用冻结的 ESM2 per-residue embedding 训练 DDQN 的可训练 MLP/Q head，在网络结构上可行；
2. checkpoint 可以严格加载回同构 DDQN Q network；
3. Round 0 全扫描同时包含正、负和近零动作，适合学习 state 内动作排序，而不是只学习正样本分类；
4. 现有 ranking + regression 组合目标和 state-macro 指标与本项目目标一致；
5. 数据生成与 RL 主代码已经分离，适合独立使用 CPU 资源继续构造标签。

#### 65.2 当前不宜直接开始正式训练的原因

1. 没有 release；
2. 没有 validation/test action labels；
3. 单 seed 数据不能提供真实 LCB；
4. 大量近零动作被浮点数符号错误地当作确定正/负样本；
5. 巨型 CSV 的双重全量内存加载不适合正式规模；
6. 只有 WT depth=0 状态，尚未覆盖 RL 会遇到的连续多突变 state；
7. `train_sft.sh` 默认 Python 路径被不存在的环境覆盖；
8. SFT checkpoint 尚未通过显式参数接入 RL 的 A/B/C/D 对照实验。

#### 65.3 建议的最小闭环

1. 修正 `train_sft.sh` 的 Python 路径覆盖；
2. 至少为 validation split 生成动作标签；test 标签保留到最终无偏评估；
3. 根据重复 repack/relax 的实测波动确定非零 noise floor，将 `|reward| <= epsilon` 设为 uncertain；
4. 采用两阶段重复策略，避免把 1187 万动作全部重复：先单 seed 扫描，再对 top candidate、near-zero 和 hard negative 使用 3--5 个 seed；
5. 重新聚合 mean/std/LCB/confidence，生成带 hash 的正式 release；
6. 将 release 按 split 分文件并采用 Parquet/分块读取，或预先缓存 ESM2 embedding；
7. 先做小规模 overfit 和 Round 0 pilot，确认 NDCG、positive@K、greedy regret 优于随机基线；
8. 再按 depth 2/4/8 逐步加入 beam-search state，降低 WT 单点分布与在线 RL 状态分布的差异；
9. 最后加入显式 `--sft-checkpoint` 等 RL 参数，完成 random initialization 与 SFT initialization 的严格对照。

最终判断：**当前代码证明了 SFT Q-head 路径在工程和架构上成立，`sft_v001` 也完成了高覆盖率的 Round 0 训练标签；但 validation、真实重复 LCB、release 和大规模 I/O 尚未闭环。当前最合理的下一步不是直接启动长时间 SFT，而是先完成可验证的数据 release，再进行小规模 pilot。**

## 2026-10-12：SFT Base Release、Dead Zone 与延迟加载实现

### 66. 本轮问答和决策记录

本轮对上一阶段报告中的五个问题进行了进一步对齐。

#### 66.1 数据集划分

需要先修正一个名词层面的理解：当前 `split_manifest.csv` 并不是直接复制 RL 的
train/validation/test。其源数据来自 RL 的 `train_index.txt`，RL validation index 被保留在 SFT
数据生成之外；随后 `prepare-manifest` 已经在这批 RL-train 来源数据内部，按照 MMseqs cluster
产生了 manifest train/validation/test。

当前真正的问题是：`run_round0.sh` 只对 manifest 的 `train` split 执行了 PyRosetta 全扫描，
所以只有这 6145 个 state 具有 action/reward label，manifest validation/test 虽然有 state 和序列，
但没有 action label。

最终严格方案应当是继续生成 manifest validation/test 的动作标签，并直接使用现有 cluster split。
考虑到重新扫描还需要数天，本轮 Base Version 采用可立即执行的 pilot 方案：

1. 只使用当前已有 label 的 manifest-train states；
2. 再按照 `cluster_id` 进行一次 80%/10%/10% 嵌套划分；
3. 新 split 用作 SFT train/validation/test；
4. 原始 manifest split 保存为 `manifest_split`，防止混淆；
5. 原始 RL validation 数据仍完全不参与 SFT。

该划分适合 Base Version 的模型选择和流程验证，但不能替代未来在原 manifest validation/test 上的
严格外部评估。

#### 66.2 SFT 训练目录

确定不再把 10 GB `labels.csv` 原样复制成 `actions.csv`。release 改为 ID-indexed、state-level
compact 格式；数据生成中间文件和正式训练 release 保持分离。

#### 66.3 近零样本

采用默认 `tau=0.01` 的 dead zone。这里不是断言所有近零值都是浮点错误，而是认为单 seed、零噪声
边界下的符号不足以构成可信正负监督。规则为：

```text
confirmed_positive: reward_lcb > +0.01
confirmed_negative: reward_ucb < -0.01
uncertain:           其他有限且非结构失败的动作
```

uncertain 动作不会删除，其 SFT regression target 被映射到0，confidence weight 默认乘以0.25；
原始 `reward_mean`、strength delta 和 toughness delta 仍完整保存，便于统计分析和指标计算。

#### 66.4 大数据读取

确定采用三层拆分：

1. `states.csv` 只保存 state ID、split、sequence 和相对文件路径；
2. 每个 state 的 `[L, 20]` target/mask/component arrays 单独保存为压缩 `.npz`；
3. 每个 state 的 ESM2 `[L, embedding_dim]` 单独保存为 `.npy`。

Dataset 初始化时只读取轻量 ID/path index；真正的 target 和 embedding 由 DataLoader worker 在
`collate_sft_states` 中按 batch 延迟读取。父进程不再加载千万行动，也不再保存全部 state tensor。

#### 66.5 SFT 与 RL 的关系

继续采用 SFT + RL。Round-0/浅层 beam 数据用于让 Q head 先学习局部 reward landscape，RL 再从
该初始化探索没有被离线 beam 覆盖的连续多突变状态。SFT-only greedy/beam 应保留为实验基线；
若未来 SFT-only 与 SFT+RL 表现相同，再据实判断 RL 是否提供额外收益。

### 67. 新的 compact release 格式

新增 `compact_state_v1`：

```text
release/sft_actions_round0_base/
├── states.csv
├── dataset_version.json
├── release_build_config.json
├── state_data/
│   └── <hash-prefix>/<state-hash>.npz
└── embeddings/
    └── esm2_1280/<hash-prefix>/<state-hash>.npy
```

`states.csv` 是轻量索引，包含：

- `state_id`、`protein_id`、`cluster_id`；
- SFT pilot `split` 和原始 `manifest_split`；
- sequence、visited positions；
- `state_data_path`、`embedding_path`、embedding dimension/dtype。

每个 state `.npz` 包含形状为 `[L,20]` 的：

- `targets`：应用 dead zone 后的 SFT target；
- `reward_means`：未经 dead-zone 归零的原始 mean；
- `confidence_weights`；
- `label_codes`；
- `observed_mask`；
- `strength_deltas`、`toughness_deltas`。

固定 amino-acid action 顺序不再在每一行重复存储，因此大幅减少 state ID、sequence、label string
等 CSV 重复字段。`release_build_config.json` 保存源文件、split seed、tau 和 target column；如果对
同一输出目录使用不同参数重跑，程序会拒绝复用旧 state 文件，避免 silent data drift。

### 68. 流式 release 构建

`sft_data_generation/release.py` 新增：

- `nested_split_states`：按 cluster 对已有标签的 manifest split 做 pilot 嵌套划分；
- `create_compact_release`：按 chunk 流式读取大型 label CSV；
- 跨 chunk 的 state group 拼接和非连续 state 检查；
- action-index 范围和重复检查；
- dead-zone 重分类、target 零中心化和 uncertain 权重缩放；
- 每 state 原子写入和中断后同配置复用；
- split cluster leakage、三 split 覆盖和 label count 检查。

默认只读取 label CSV 中实际需要的列，而不是 materialize 约10 GB的完整 DataFrame。

同时修正 `build-tasks --split` 的 CLI 默认行为。之前 `--split validation` 会在默认 `train` 后继续
append；现在未指定时才使用 train，显式指定 validation/test 时只生成所要求的 split。

### 69. ESM2 embedding 独立缓存

新增 `sft_data_generation/embeddings.py` 和 CLI `cache-embeddings`：

1. 从 compact `states.csv` 读取 sequence；
2. 使用现有 `ESM2SequenceEncoder.encode_sequences` 进行变长 batch 推理；
3. 每个 state 独立保存 `[L,1280]` embedding；
4. 默认使用 float16 降低磁盘与 DataLoader I/O；
5. 对已有 cache 检查 shape 和 dtype，可断点复用；
6. 原子写入，全部完成后更新 `states.csv` 和 `dataset_version.json`；
7. 训练前可检查 cache 是否 complete、dimension 是否与 Q head 一致。

ESM2 embedding 在 SFT 训练期间保持冻结；训练过程不再每个 epoch 重复执行 ESM2 forward。

### 70. ID-based Dataset 与 collate 延迟读取

`model/sft_module/dataset.py` 新增 `SFTStateReference`。对于 compact release：

- `SFTActionDataset.__init__` 只构建 state ID/path reference；
- `__getitem__` 返回 reference，不读取 `.npz` 或 `.npy`；
- `collate_sft_states` 收到一个 batch 的 reference 后再加载 target 和 embedding；
- 根据 batch 最大长度做 padding；
- 继续生成 observed/valid/positive/negative/visited masks；
- legacy `states.csv + actions.csv` 仍保留兼容读取路径。

`model/sft_module/trainer.py` 优先使用 batch 中缓存的 `embeddings`；只有显式关闭 cached embedding
要求且使用 legacy release 时，才回退到在线 ESM2 encoder。

### 71. 训练入口修改

`train_sft.py` 增加：

- `--require-cached-embeddings/--no-require-cached-embeddings`，默认要求 cache；
- train、validation、test 三个 ID-based dataset；
- validation macro NDCG 选择 best checkpoint；
- 训练结束后重新加载 `best.pt`，只在此时评估一次 test split；
- test loss、NDCG、Spearman、positive@K、greedy reward 等写入日志和 TensorBoard；
- `run_summary.json` 保存独立 test metrics。

`train_sft.sh` 已修正重复 `PYTHON_BIN` 覆盖问题，默认 Python 恢复为：

```text
/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python
```

默认 release 修改为 `sft_actions_round0_base`。preflight 不再要求训练期间加载 ESM checkpoint，
而是检查：

- release format 为 `compact_state_v1`；
- embedding cache 已完成；
- cached embedding dimension 与 SFT Q head 一致；
- PyTorch/CUDA 和 Q-head architecture contract 正确。

本轮没有修改 RL environment、replay、agent 更新逻辑或 RL 启动脚本。

### 72. 使用方式

一键构建 compact release 并缓存 ESM2：

```bash
bash sft_data_generation/scripts/prepare_compact_release.sh
```

可通过环境变量覆盖关键参数：

```bash
SFT_ACTIONS=/path/to/labels.csv \
SFT_RELEASE_NAME=sft_actions_round0_base \
SFT_EMBEDDING_DEVICE=cuda:0 \
SFT_EMBEDDING_BATCH_SIZE=4 \
SFT_EMBEDDING_DTYPE=float16 \
bash sft_data_generation/scripts/prepare_compact_release.sh
```

正式训练：

```bash
bash train_sft.sh
```

构建脚本分为 `release-compact` 和 `cache-embeddings` 两个显式 CLI stage，因此10 GB labels 转换和
GPU ESM2 推理可以分别运行和恢复。

### 73. 测试结果和当前边界

完成以下验证：

```text
Python compile checks: passed
shell syntax checks: passed
compact nested cluster split: passed
chunk-boundary state reconstruction: passed
tau=0.01 dead-zone classification: passed
per-state NPZ target export: passed
per-state NPY embedding cache/reuse: passed
ID-only Dataset + collate lazy loading: passed
one-epoch cached-embedding train/validation/test smoke test: passed
SFT/DDQN strict architecture compatibility: passed
focused SFT + DDQN + dataset tests: 52 passed in 3.34 s
```

测试使用 synthetic compact release，没有启动真实 10 GB labels 的完整转换，也没有执行 6145 个
state 的真实 ESM2 缓存。这两个步骤会产生大量磁盘 I/O 和 GPU 计算，应通过上述脚本作为独立长任务
运行。当前 Base Version 的 test split 是从已有标签的 manifest-train clusters 中再次划分出来的
pilot test；后续应为原 manifest validation/test 生成标签，构造严格最终 release。

## Plan B：SFT `best.pt` 初始化 DDQN

### 74. 修改前检查结论

修改前，`train_version_terminal_async.sh` 和 `asynchronous_training.py` **不支持**使用 SFT
`best.pt` 初始化 DDQN：learner 直接构造随机 Q head，随后将该随机 online network 作为
`initial_policy_state` 发给所有 actors。

本次使用的 checkpoint 为：

```text
/mnt/nas/jianquanzhao/data/mprl/SFT/model/sft_qhead_v001_20261014_112424/best.pt
```

其 architecture metadata 与当前异步训练配置一致：

```text
embedding_dim: 1280
visited-mask feature: enabled
hidden_dims: [256, 256]
per-residue action dimension: 20
amino-acid order: ACDEFGHIKLMNPQRSTVWY
```

### 75. 最小实现方案

`training.py` 新增 `--sft-checkpoint`（别名 `--sft-q-head-checkpoint`）和严格加载函数：

1. 检查文件存在且 `checkpoint_type=mechanical_protein_sft_q_head`；
2. 严格核对 embedding dimension、hidden dimensions、visited mask、氨基酸顺序和输出维度；
3. 拒绝包含 NaN/Inf 或缺失参数的 checkpoint；
4. 用 `strict=True` 将 `q_network_state_dict` 加载到 online Q network；
5. 立即 hard-sync 到 target Q network；
6. 不加载 SFT optimizer/scheduler，不恢复 replay、epsilon 进度或 DDQN 计数器；
7. 禁止同时使用 `--sft-checkpoint` 与 `--resume-checkpoint-dir`，避免混淆“新实验初始化”和
   “恢复既有 RL 训练”。

`asynchronous_training.py` 在 learner agent 创建后、replay 创建及 worker 启动前执行上述加载。
因此首次传给 actors 的 `initial_policy_state` 已经是 SFT 权重，后续 actor policy broadcast 和
普通异步 DDQN 完全沿用原逻辑。

同步 `training.py` 入口也复用了同一个加载函数；没有修改 Q network、loss、PER、n-step、
target sync 或异步调度逻辑。

### 76. 启动方法

Plan B（当前 `best.pt` initialization + DDQN）：

```bash
MPRL_SFT_CHECKPOINT=/mnt/nas/jianquanzhao/data/mprl/SFT/model/sft_qhead_v001_20261014_112424/best.pt \
bash train_version_terminal_async.sh
```

脚本会在自动生成的 run label 后添加 `_sft`，stdout 启动摘要也会打印 checkpoint 路径。
如需自定义实验名：

```bash
MPRL_RUN_LABEL=plan_b_sft_ddqn \
MPRL_SFT_CHECKPOINT=/mnt/nas/jianquanzhao/data/mprl/SFT/model/sft_qhead_v001_20261014_112424/best.pt \
bash train_version_terminal_async.sh
```

不设置 `MPRL_SFT_CHECKPOINT` 时仍为 Plan A 的 random Q initialization，便于保持 A/B 对照。
也可以直接在 Python 命令中传入：

```text
--sft-checkpoint /mnt/nas/jianquanzhao/data/mprl/SFT/model/sft_qhead_v001_20261014_112424/best.pt
```

实际运行日志中应出现：

```text
Initialized DDQN online/target Q heads from SFT checkpoint ...
optimizer/replay/counters remain fresh
```

`run_config.json` 的 `args.sft_checkpoint` 会保存本次使用的绝对/传入路径，便于追溯。

### 77. 验证结果

完成以下验证：

```text
Python compile checks: passed
shell syntax check: passed
CLI --help exposure: passed
missing checkpoint fail-fast: passed
focused DDQN/SFT/training/asynchronous tests: 62 passed, 2 skipped
real best.pt strict load: passed
real best.pt -> asynchronous ActorPolicy handoff/greedy action probe: passed
```

真实 checkpoint 的 CPU 探针结果：

```text
checkpoint epoch: 4
11-residue batched output shape: (2, 220)
online-target maximum absolute difference: 0.0
optimizer state entries: 0
environment steps: 0
optimization steps: 0
actor greedy action for 11 residues: 15 (valid range 0-219)
```

因此当前 checkpoint 可以直接用于 Plan B。其 SFT validation `NDCG=0.6614`，但
`Spearman=0.0923`、greedy reward median 为 `0`；这意味着该初始化更适合作为改善早期动作排序和
正动作命中率的 warm start，不能视为已经解决长期多步策略问题。正式对比时仍应在相同 seed、数据划分、
环境步数和 validation PDB 上比较 Plan A 与 Plan B 的 greedy validation reward、
strength/toughness delta、positive fraction 和学习速度。

## 78. Plan D：SFT proposal 与 online-Q mixture 实现方案

### 78.1 核心原则

Plan D 不应只是再次使用 `best.pt` 初始化 DDQN，而是需要同时保留两套 Q head：

- `Q_sft`：从 `best.pt` 加载并全程冻结，只负责提供动作 proposal；
- `Q_online`：可以同样从 `best.pt` 初始化，但继续接受 DDQN 的在线 TD 更新；
- `Q_target`：继续按照现有 DDQN 逻辑从 `Q_online` 同步，不与 `Q_sft` 混合。

这样，即使在线 TD 更新逐渐覆盖了初始化参数，actor 仍然可以访问未经修改的原始 SFT 策略。

### 78.2 推荐的第一版：首步硬混合

当前 SFT 数据全部来自 `depth=0` 状态，因此第一版建议只让 SFT 参与每个 episode 的第一个动作：

```text
rho(t) = 0.5 -> 0.0，线性退火

if validation:
    action = argmax Q_online
elif episode_step == 0 and random() < rho(t):
    action = sample_from_topk(Q_sft, action_mask)
else:
    action = epsilon_greedy(Q_online, epsilon)
```

对应的动作来源概率为：

```text
P(SFT proposal)      = rho
P(random exploration) = (1-rho) * epsilon
P(online greedy)       = (1-rho) * (1-epsilon)
```

实现时必须先判断 SFT proposal，再执行 epsilon-greedy。如果先判断 epsilon，那么训练初期
`epsilon=1` 时，SFT 被选中的概率将变为零，无法达到改善早期 replay 的目的。

建议第一轮参数为：

```bash
--sft-checkpoint best.pt
--sft-proposal-checkpoint best.pt
--sft-proposal-start 0.50
--sft-proposal-end 0.0
--sft-proposal-decay-steps 369648
--sft-proposal-top-k 5
--sft-proposal-temperature 1.0
--sft-proposal-max-depth 1
```

其中 `369648 = 7701 proteins * 24 steps * 2 epochs`，表示在大约两个 epoch 内将 proposal
比例从 50% 线性退火到零。SFT proposal 不建议固定选择 top-1；在有效动作的 top-5 内按照
softmax 或排名进行采样，可以增加轨迹多样性，避免所有 actor 集中产生少数相同轨迹。

### 78.3 其他可行方案及优缺点

| 方案 | 实现方式 | 优点 | 缺点 |
|---|---|---|---|
| 首步硬混合 | SFT 只参与 `episode_step=0` | 与 depth=0 训练分布一致；因果解释清楚；风险最低 | SFT 对后续轨迹的直接影响有限 |
| 全程硬混合 | 每一步都以 `rho(t)` 选择 SFT 或 online Q | SFT 影响更强；可能更快提高早期 replay 质量 | 多突变状态对 SFT 属于 OOD，可能产生错误 proposal |
| Episode/actor 分流 | 一部分 episode 或 actor 使用 SFT，其余使用 online Q | 异步架构中实现简单；便于比较不同策略产生的数据 | 整条 SFT 轨迹受到 depth 偏移影响；actor 间数据相关性更强 |
| Q 值融合 | 使用标准化后的 `alpha*Q_sft + (1-alpha)*Q_online` | 动作变化平滑；每一步综合两种判断 | 两套 Q 的尺度不同，不能直接相加；需要逐状态校准 |
| 排名融合 | 对两套网络给出的 action rank 加权 | 不受 Q 值绝对尺度影响；通常比数值融合稳定 | 会丢失动作间分数差异；实现和解释更复杂 |
| SFT 候选、online 重排 | SFT 产生 top-k，online Q 在候选中选择 | 限制探索空间；可逐渐将控制权交给 online Q | 初期 online Q 可能把 SFT 的优质动作重新排到后面 |
| SFT 替代部分随机探索 | epsilon 分支中用 SFT top-k 替代部分随机动作 | 修改量最小；容易接入现有代码 | SFT 使用率被 epsilon 绑定，无法独立控制和评估 |

首轮实验应优先采用“首步硬混合”。如果它能提升 replay 中的正向样本比例，但复杂状态中的后续动作
仍然较差，再补充 depth 1/2/4 数据，并逐步扩展为多步或全程混合。

### 78.4 对当前代码的最小修改位置

`training.py`：增加 proposal checkpoint、起止比例、退火步数、top-k、temperature 和最大适用
depth 参数。原有 `--sft-checkpoint` 继续负责初始化 DDQN online/target Q head。

`asynchronous_training.py`：加载并严格校验冻结的 SFT state dict，在 actor 启动时发送一次静态
SFT 策略。后续的 policy broadcast 只更新 `Q_online`，不更新 `Q_sft`。

`model/asynchronous_module/runtime.py`：让 `ActorPolicy` 同时持有 `online_network` 和
`sft_proposal_network`。硬混合分支只运行被选中的网络，以避免不必要的双重前向计算。所有动作都必须
使用相同的 `action_mask`，包括 no-op、不可突变位置和已访问位置约束。

动作选择函数除返回 action 外，还应记录：

```text
action_source = sft_topk / online_greedy / random
proposal_ratio
sft_rank
episode_step
```

ReplayBuffer 的 Bellman 学习逻辑不需要改变。DDQN 本身是 off-policy 算法，可以使用混合行为策略产生的
transition；PER 的 importance-sampling correction 仍然只负责修正 replay 抽样概率。

### 78.5 日志与 TensorBoard 指标

至少增加以下指标：

```text
policy/sft_action_fraction
policy/online_greedy_fraction
policy/random_action_fraction
policy/proposal_ratio
reward/terminal_by_first_action_source
reward/positive_fraction_by_first_action_source
replay/sft_source_fraction
```

还应分别比较由 SFT 首动作和非 SFT 首动作启动的 episode 的 terminal reward、strength delta、
toughness delta 和 positive fraction，以验证 SFT proposal 是否真的提高了后续轨迹质量。

### 78.6 Validation 设计

固定 greedy validation 的主指标必须只评估在线策略：

```text
epsilon = 0
rho = 0
action = argmax Q_online
```

否则只能证明“混合行为策略效果更好”，不能证明 DDQN 已经学会并保留了 SFT 带来的优势。可以额外记录
hybrid validation，但不能用它替代 online-only greedy validation。

建议在相同数据划分、seed、环境步数和验证 PDB 上比较：

```text
A: random initialization + current DDQN
B: SFT initialization + current DDQN
D1: SFT initialization + first-step SFT top-k proposal
D2: SFT initialization + all-step SFT proposal（后续扩展）
```

### 78.7 成功判据与停止条件

Plan D 的成功不能只依据 proposal 生效期间 actor reward 上升，而应同时满足：

1. proposal 生效期间，replay 中正向 terminal transition 的比例和质量提高；
2. `rho` 退火到零后，online-only fixed greedy validation 仍保留提升；
3. strength 和 toughness 没有出现一项改善、另一项严重下降；
4. 多个随机 seed 下方向一致，bootstrap confidence interval 支持改善。

如果 proposal 生效期间 reward 提高，但 `rho=0` 后立即回落，说明 SFT 只改善了行为策略，DDQN 并未
成功吸收这些轨迹。届时应考虑增加辅助 ranking/SFT loss、降低早期 online learning rate、短期冻结部分
Q head，或调整 replay 中 demonstration 的保留方式，而不是简单延长训练时间。

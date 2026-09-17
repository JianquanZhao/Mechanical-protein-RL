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

# DDQN + ESM2 编码空间更新说明

## 依赖安装

已在 `mprl-vgpt` 环境中安装用户指定依赖：

```bash
python -m pip install fair-esm
```

验证结果：

```text
fair-esm 2.0.0
esm /home/jianquanzhao/anaconda24/envs/mprl-vgpt/lib/python3.10/site-packages/esm/__init__.py
```

说明：`pip show fair-esm` 会被用户 site-packages 中损坏的 `grpcio` metadata 干扰；使用 `PYTHONNOUSERSITE=1` 后可正常通过 Python metadata 和 `import esm` 验证。

## 核心设计

原始 `QNetwork` 直接将环境 observation flatten 后接多层 Linear，这会把网络输入维度绑定到单个蛋白的长度，例如默认 one-hot observation 是 `L * 20`。为了后续支持多个不同长度蛋白，本次将 DDQN Q 网络改为面向“蛋白语言模型编码空间”的 Q head。

新增支持的编码空间维度：

```text
1280, 2560, 5120
```

默认值：

```text
1280
```

## model/agent_module/ddqn_agent.py

新增常量：

```python
SUPPORTED_PROTEIN_EMBEDDING_DIMS = (1280, 2560, 5120)
```

`DDQNConfig` 新增字段：

```python
embedding_dim: int = 1280
```

并在 `validate()` 中限制其必须为 `1280/2560/5120`。

`QNetwork` 的结构从：

```text
flatten(state) -> Linear hidden stack -> action_dim
```

调整为：

```text
flatten(state)
    -> input_projection 到 embedding_dim
    -> Linear hidden stack
    -> action_dim
```

如果输入本身已经是 `(embedding_dim,)`，则 `input_projection` 使用 `nn.Identity()`；如果仍然是当前项目已有的 one-hot observation，例如 `(3460,)`，则使用 `nn.Linear(flattened_dim, embedding_dim)` 投影到蛋白编码空间。

这样做的好处：

- 当前已有训练和测试不需要立即切换到 ESM2，也能继续运行；
- 后续使用 ESM2 pooled embedding 时，`state_shape=(embedding_dim,)`，网络自然变成纯 Q head；
- checkpoint 中会保存新的 `embedding_dim` 配置。

## model/encoding_module

新增模块：

```text
model/encoding_module/__init__.py
model/encoding_module/esm2_encoder.py
```

新增 `ESM2SequenceEncoder`，用于将环境当前 protein sequence 编码为固定维度 ESM2 embedding。

维度与模型映射：

```text
1280 -> esm2_t33_650M_UR50D
2560 -> esm2_t36_3B_UR50D
5120 -> esm2_t48_15B_UR50D
```

支持 pooling：

```text
mean: 对 residue token embedding 做平均，默认
cls: 使用 BOS/CLS token 表示
```

注意：该 encoder 只有在显式构造时才会加载 ESM2 模型和权重。默认训练不会自动下载大模型。

## training.py

新增训练参数：

```bash
--embedding-dim {1280,2560,5120}
--observation-encoder {default,esm2}
--esm2-device ESM2_DEVICE
--esm2-pool {mean,cls}
--esm2-mutable-only
```

默认行为：

```bash
python training.py
```

仍使用项目原有 one-hot observation，但 DDQN QNetwork 会先投影到默认 `1280` 维编码空间。

使用 ESM2 observation：

```bash
python training.py \
  --observation-encoder esm2 \
  --embedding-dim 1280 \
  --esm2-device cuda:0 \
  --esm2-pool mean
```

更大 ESM2 编码空间：

```bash
python training.py \
  --observation-encoder esm2 \
  --embedding-dim 2560
```

或：

```bash
python training.py \
  --observation-encoder esm2 \
  --embedding-dim 5120
```

注意：`2560/5120` 对应的 ESM2 模型非常大，首次运行会下载权重，并且显存/内存压力会明显高于 1280。

## model/agent_module/__init__.py

导出了：

```python
SUPPORTED_PROTEIN_EMBEDDING_DIMS
```

方便后续训练脚本或配置系统复用合法维度集合。

## tests/test_ddqn_agent.py

新增测试覆盖：

- 默认 `QNetwork.embedding_dim == 1280`；
- `QNetwork` 支持 `1280/2560/5120` 三种蛋白编码维度；
- legacy observation 会通过 `input_projection` 投影到编码空间；
- `DDQNConfig.embedding_dim` 非法值会触发校验错误。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/agent_module/ddqn_agent.py \
  model/agent_module/__init__.py \
  model/encoding_module/__init__.py \
  model/encoding_module/esm2_encoder.py
```

DDQN 单测：

```text
24 passed in 2.54s
```

默认训练冒烟测试：

```bash
python training.py \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-ddqn-embedding-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --checkpoint-every 1 \
  --no-resume-logs \
  --log-level WARNING
```

结果：成功完成 1 个 episode。

全量测试：

```text
76 passed in 8.08s
```

## 当前边界

本次修改已经让 DDQN Q head 面向固定维度蛋白语言模型 embedding，并提供 ESM2 observation encoder。但如果要在同一个 replay buffer / 同一个 agent 中混合不同长度蛋白，还需要进一步处理 `action_dim = L * 20` 随蛋白长度变化的问题。后续更彻底的做法是将 Q head 改成 per-residue 输出：

```text
ESM2 per-residue embedding [L, D] -> Q [L, 20] -> action mask
```

这会比当前 pooled embedding 更适合真正的多蛋白、多长度联合训练。

---

# 追加更新：ESM2 per-residue Q head

本次继续完成了从 pooled/fixed embedding Q head 到 per-residue Q head 的改造。

## 核心变化

原先 ESM2 observation encoder 输出 pooled embedding：

```text
ESM2 token representations [L, D] -> mean/cls pooling -> [D]
DDQN QNetwork [D] -> [action_dim]
```

现在改为 per-residue 输出：

```text
ESM2 residue representations [L, D]
    -> residue-wise Q head
    -> [L, 20]
    -> flatten
    -> [L * 20]
```

其中 `20` 对应 20 种 canonical amino acids，和环境原有 action 编码保持一致：

```text
action = mutable_position_index * 20 + amino_acid_index
```

## model/agent_module/ddqn_agent.py

新增常量：

```python
AMINO_ACID_ACTION_DIM = 20
```

`QNetwork` 现在有两条路径：

1. Legacy / pooled 路径：

```text
state [D] or legacy flattened one-hot
    -> input_projection
    -> MLP
    -> [action_dim]
```

2. Per-residue 路径：

```text
state [L, embedding_dim]
    -> shared residue_head
    -> [L, 20]
    -> flatten
    -> [L * 20]
```

当 `state_shape == (L, embedding_dim)` 时会进入 per-residue 模式，并强制：

```text
action_dim == L * 20
```

如果不满足会直接报错，避免 ESM2 per-residue 输入被错误地走 flatten fallback。

`DDQNAgent` 同步支持了 per-residue state：

- 单状态 action selection 可根据当前 state 的 residue 数计算 action mask 长度；
- `action_mask` 长度必须等于 `current_L * 20`；
- batch optimize 仍要求 batch 内 state shape 一致，因为当前 ReplayBuffer 仍是 NumPy 固定数组存储。

## model/encoding_module/esm2_encoder.py

`ESM2SequenceEncoder` 不再支持 mean/cls pooling，直接返回 residue token embeddings：

```python
array.shape == (len(sequence), embedding_dim)
```

即：

```text
[L, D]
```

同时，`mutable_only` 默认改为 `True`。原因是 DDQN action space 是按 `mutable_positions` 排列的，per-residue Q 输出的第 `i` 行必须对应第 `i` 个 mutable position。

如果后续需要“全序列上下文 + 只对 mutable positions 输出 Q”，需要进一步实现 position mapping 或 contextual gather 逻辑。当前版本优先保证 action 轴严格对齐。

## training.py

删除了旧的 pooling 参数：

```text
--esm2-pool
```

保留并更新了：

```bash
--observation-encoder {default,esm2}
--embedding-dim {1280,2560,5120}
--esm2-device ESM2_DEVICE
--esm2-mutable-only
--esm2-full-sequence
```

默认 ESM2 行为现在是：

```text
encode mutable sequence -> [num_mutable_positions, embedding_dim]
```

使用方式：

```bash
python training.py \
  --observation-encoder esm2 \
  --embedding-dim 1280 \
  --esm2-device cuda:0
```

如果显式使用：

```bash
--esm2-full-sequence
```

则需要确保 full sequence residue 数和环境 action positions 数一致，否则 `QNetwork` 会因为 `action_dim != L * 20` 报错。

## tests/test_ddqn_agent.py

新增测试覆盖：

- `QNetwork((L, 1280), action_dim=L*20)` 会进入 per-residue 模式；
- batch 输入 `[B, L, D]` 输出 `[B, L*20]`；
- 单状态输入 `[L, D]` 输出 `[1, L*20]`；
- per-residue 输入下 `action_dim != L*20` 会报错。

## 验证结果

编译检查通过：

```bash
python -m py_compile \
  training.py \
  model/agent_module/ddqn_agent.py \
  model/encoding_module/esm2_encoder.py \
  tests/test_ddqn_agent.py
```

DDQN 单测：

```text
26 passed in 2.40s
```

默认训练冒烟测试：

```bash
python training.py \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-per-residue-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --checkpoint-every 1 \
  --no-resume-logs \
  --log-level WARNING
```

结果：成功完成。

全量测试：

```text
78 passed in 7.84s
```

## 当前边界

这次已经完成了 per-residue Q head 和 ESM2 per-residue observation 输出。但当前 `ReplayBuffer` 仍然使用固定 shape 的 NumPy 数组，因此：

- 同一个 replay batch 内的蛋白长度必须一致；
- 如果要把不同长度蛋白混在同一个 replay buffer 中，需要继续将 ReplayBuffer 改成 ragged/padded batch 形式；
- 对应的 action masks 也需要 padding 到 batch 内最大 `L * 20`，并确保 padded actions 永远 masked out。

也就是说，本次完成的是网络和 observation 侧的 per-residue 化；真正的多长度混合 replay 还需要下一步改造 ReplayBuffer 和 batch collation。

---

# 追加更新：variable-length ReplayBuffer 与 padded batch collation

本次完成了 ReplayBuffer 和 DDQN batch 侧的多长度混合支持。目标是让不同长度蛋白的 per-residue ESM2 observation 可以进入同一个 replay buffer，并在采样时 collate 成同一个 padded batch。

## 核心设计

对 per-residue ESM2 observation：

```text
state_i: [L_i, D]
action_mask_i: [L_i * 20]
```

ReplayBuffer 采样一个 batch 时，取本批次最大长度：

```text
L_max = max(L_i)
```

然后 padding 为：

```text
states:            [B, L_max, D]
next_states:       [B, L_max, D]
action_masks:      [B, L_max * 20]
next_action_masks: [B, L_max * 20]
```

padding 规则：

- state / next_state 的 padded residue embedding 填 0；
- action mask / next action mask 的 padded action 位置填 `False`；
- 因此 DDQN 不会选择或 bootstrap 到 padded residue 的动作。

## model/replay_buffer_module/replay_buffer.py

`ReplayBuffer` 新增参数：

```python
variable_length: bool = False
```

默认保持原有固定 shape NumPy 预分配行为，兼容已有 one-hot / 单长度训练。

当 `variable_length=True` 时：

- 内部 `states` / `next_states` 使用 object slots 保存不同长度数组；
- `action_masks` / `next_action_masks` 也使用 object slots 保存不同长度 mask；
- `add()` 时允许第一维长度不同，但要求 trailing shape 一致，例如 `[L, 1280]` 的 `1280` 必须一致；
- 当前 transition 的 action 合法范围根据当前 state 动态计算：

```text
action_dim_i = L_i * 20
```

- `sample()` 时自动进行 padded batch collation。

新增/修改的关键方法：

```python
_action_dim_for_state(...)
_pad_state_batch(...)
_pad_mask_batch(...)
_snapshot_object_array(...)
```

序列化也同步支持 variable-length buffer：

- metadata 中新增 `variable_length`；
- object arrays 通过 `np.savez_compressed(..., allow_pickle=True load)` 保存/读取；
- 旧 snapshot 没有 `variable_length` 时按 `False` 兼容读取。

## model/agent_module/ddqn_agent.py

DDQNAgent 的 per-residue batch 校验已调整：

以前要求：

```text
batch state trailing shape == agent.state_shape
```

现在 per-residue 模式下允许 batch 的 residue 维度是 padded 后的 `L_max`：

```text
states: [B, L_max, embedding_dim]
```

只要求：

```text
states.ndim == 3
states.shape[-1] == embedding_dim
```

优化时动态计算本批次 Q/action 维度：

```text
batch_action_dim = L_max * 20
```

并要求：

```text
next_action_masks.shape == [B, batch_action_dim]
```

这使得 agent 可以：

- 用初始化时的某个蛋白长度构建网络；
- 在 replay batch 中接收 padded 到不同 `L_max` 的 batch；
- 通过 residue-wise shared head 输出 `[B, L_max * 20]`；
- 使用 padded mask 屏蔽无效动作。

## training.py

训练入口中 ReplayBuffer 构建逻辑已更新：

```python
variable_length=args.observation_encoder == "esm2"
```

也就是说：

- 默认 `--observation-encoder default`：仍使用固定长度 replay buffer；
- `--observation-encoder esm2`：自动启用 variable-length replay buffer 和 padded batch。

这保持了旧训练路径稳定，同时让 ESM2 per-residue 路径自动进入多长度兼容模式。

## tests/test_replay_buffer.py

新增测试：

- `test_variable_length_sample_pads_states_and_masks`
  - 向同一个 buffer 添加 `[2, D]` 和 `[3, D]` 两种长度；
  - sample 后检查 states padded 到 `[B, 3, D]`；
  - masks padded 到 `[B, 60]`；
  - 短序列 padded mask 区域全为 `False`。

- `test_variable_length_save_and_load_round_trip`
  - 检查 variable-length replay buffer 可保存和读取；
  - 读取后仍能 sample 出 padded batch。

## tests/test_ddqn_agent.py

新增测试：

- `test_ddqn_agent_optimizes_padded_per_residue_batch`
  - agent 初始化 state shape 为 `[2, 1280]`；
  - replay batch padded 到 `[B, 3, 1280]`；
  - next action mask 为 `[B, 60]`；
  - 优化可以正常完成。

这验证了 ReplayBuffer padded batch 和 DDQN per-residue Q head 已经在训练优化路径上对齐。

## 验证结果

编译检查：

```bash
python -m py_compile \
  model/replay_buffer_module/replay_buffer.py \
  model/agent_module/ddqn_agent.py \
  training.py \
  tests/test_replay_buffer.py \
  tests/test_ddqn_agent.py
```

ReplayBuffer + DDQN 单测：

```text
52 passed in 2.40s
```

全量测试：

```text
81 passed in 7.88s
```

默认训练冒烟测试：

```bash
python training.py \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-variable-replay-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --checkpoint-every 1 \
  --no-resume-logs \
  --log-level WARNING
```

结果：成功完成。

## 当前边界

本次已经完成：

```text
不同 L 的 transition -> 同一个 ReplayBuffer -> sample padded batch -> DDQN optimize
```

仍需注意：

- 当前训练入口本身仍是单环境循环；真正“多蛋白同时训练”还需要上层 dataset/env 切换逻辑，把不同 PDB 对应的 transition 喂进同一个 buffer。
- 一个 sampled batch 内会按 `L_max` padding，过长蛋白会增加显存占用。
- action index 仍采用每个蛋白自身的 flat index：

```text
mutable_position_index * 20 + amino_acid_index
```

因为 padded 区域 mask 为 `False`，不会参与动作选择和 target bootstrap。

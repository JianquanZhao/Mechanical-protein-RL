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

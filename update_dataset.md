# Dataset / Multi-PDB Episode 更新说明

本次修改目标：让训练不再绑定单个 `initial_pdb_path`，而是可以基于一个 PDB 文件夹，在不同 episode 中加载不同蛋白质结构，并配合前面已完成的 variable-length ReplayBuffer 进行训练。

## model/environment_module/environment.py

### MechanicalProteinEnv 初始化行为调整

原行为：

```python
env = MechanicalProteinEnv(initial_pdb_path="xxx.pdb")
```

`__init__` 立即加载 PDB、建立 reference/current pose、计算 mutable positions、创建 action space 和 observation space。

新行为：

```python
env = MechanicalProteinEnv()
```

`__init__` 只完成实例化必须的配置工作：

- 保存环境超参数；
- 初始化 backend；
- 保存 reward calculator 配置；
- 创建占位 action/observation space；
- 不要求立即加载 PDB。

同时保留兼容：

```python
env = MechanicalProteinEnv(initial_pdb_path="xxx.pdb")
```

如果仍传入 `initial_pdb_path`，会立即加载该 PDB，旧测试和旧调用方式仍可用。

### reset 支持切换 episode PDB

`reset()` 新增参数：

```python
env.reset(pdb_path="protein_a.pdb")
```

也支持 Gymnasium 风格：

```python
env.reset(options={"pdb_path": "protein_a.pdb"})
```

每次传入新的 `pdb_path` 时，环境会：

- 加载新的 PDB；
- 重建 `reference_pose` 和 `current_pose`；
- 重新计算 `total_residues`；
- 根据当前结构重新确定 `mutable_positions`；
- 重新计算 `n_actions = n_mutable_positions * 20`；
- 重建 `action_space`；
- 重建 `observation_space`；
- 重建默认 `StepRewardCalculator(reference_pose, ...)`；
- 清空 episode 状态，如 step、visited positions、history 等。

`reset()` 返回的 `info` 中新增：

```python
info["pdb_path"]
```

用于记录当前 episode 来源结构。

## model/dataset_module/dataset.py

新增 dataset 模块：

```text
model/dataset_module/__init__.py
model/dataset_module/dataset.py
```

提供：

```python
discover_structure_files(...)
ProteinStructureDataset
```

### 文件扫描

默认递归扫描：

```text
.pdb
.ent
```

### train / val 划分

使用：

```python
ProteinStructureDataset.from_folder(
    pdb_dir,
    val_fraction=0.1,
    seed=7,
)
```

会在 PDB 文件夹下生成：

```text
train_index.txt
val_index.txt
```

索引文件中保存的是相对 `pdb_dir` 的路径，便于迁移整个数据文件夹。

如果索引文件已存在，默认直接读取；如果想重新划分：

```python
ProteinStructureDataset.from_folder(..., recreate_indices=True)
```

### 训练与验证路径

Dataset 提供：

```python
dataset.train_paths
dataset.val_paths
dataset.sample_train_path(rng)
dataset.validation_paths(limit=N)
```

训练时从 `train_paths` 中采样 episode PDB；验证时按 `val_paths` 顺序运行。

## training.py

### 数据参数从 PDB 文件改为 PDB 文件夹

移除单文件入口的使用方式，新增/使用文件夹入口：

```bash
--pdb-dir PDB_DIR
--train-index TRAIN_INDEX
--val-index VAL_INDEX
--val-fraction VAL_FRACTION
--dataset-seed DATASET_SEED
--recreate-splits
```

默认：

```bash
--pdb-dir model/reward_module
```

### 训练流程变化

训练启动时：

1. 构建 `ProteinStructureDataset`；
2. 生成或读取 train/val index；
3. 用训练集第一个 PDB 初始化 env/agent/replay buffer 的基本 shape；
4. 每个 episode 从训练集采样一个 PDB；
5. 调用：

```python
env.reset(pdb_path=str(episode_pdb_path))
```

6. 正常执行 DDQN 训练；
7. episode log 中记录：

```python
"source_pdb": ...
```

### 验证流程

新增验证参数：

```bash
--validate-every VALIDATE_EVERY
--validation-episodes VALIDATION_EPISODES
```

当满足周期时，训练脚本会：

- 从 val index 中取最多 `validation_episodes` 个 PDB；
- 使用 `evaluate=True` 的 greedy action；
- 不写入 replay buffer；
- 不更新 epsilon schedule；
- 将结果追加写入：

```text
outputs/.../logs/validation.jsonl
```

每条记录包含：

```text
episode
validation_index
pdb_path
total_reward
steps
sequence
```

### ReplayBuffer 连接

当使用：

```bash
--observation-encoder esm2
```

训练入口会自动创建：

```python
ReplayBuffer(..., variable_length=True)
```

这样不同长度蛋白的 per-residue ESM2 observation 可以进入同一个 replay buffer，并在 sample 时 padded batch。

默认 one-hot observation 仍然使用固定长度 replay buffer。因此如果 PDB 文件夹中蛋白长度不同，建议使用：

```bash
--observation-encoder esm2
```

## tests

新增：

```text
tests/test_dataset.py
```

覆盖：

- 递归发现 `.pdb/.ent`；
- 生成 train/val index；
- 再次读取已存在 index；
- split 数量符合预期。

修改：

```text
tests/test_environment.py
```

新增测试：

- `MechanicalProteinEnv()` 不传初始 PDB；
- `reset(pdb_path="short.pdb")` 加载短序列；
- `reset(pdb_path="long.pdb")` 替换为长序列；
- action space 和 observation shape 随 episode PDB 更新。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/environment_module/environment.py \
  model/dataset_module/__init__.py \
  model/dataset_module/dataset.py \
  tests/test_environment.py \
  tests/test_dataset.py
```

环境与 dataset 测试：

```text
13 passed in 0.24s
```

全量测试：

```text
84 passed in 7.85s
```

文件夹训练冒烟测试：

```bash
python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-dataset-folder-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --checkpoint-every 1 \
  --no-resume-logs \
  --validate-every 0 \
  --log-level WARNING
```

结果：成功完成。

## 当前边界

- 当前训练入口是单环境循环，每个 episode 采样一个 PDB；不是并行多环境采样。
- 默认 one-hot observation 仍要求固定长度；多长度训练建议使用 ESM2 per-residue observation。
- 如果显式传入固定 `mutable_positions`，不同 PDB 必须都包含这些 pose positions；否则 reset 会报错。
- validation 当前只运行 greedy rollout 并记录 JSONL，还没有独立 checkpoint selection / early stopping。

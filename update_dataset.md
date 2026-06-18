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

# 2026-06-18 蛋白结构数据预处理

## 背景

用户的数据文件夹中可能同时包含蛋白质、DNA/RNA 或其他生物分子结构文件。训练入口虽然使用 `--pdb-dir` 指向整个结构文件夹，但 DDQN 环境当前只支持 canonical amino-acid mutation，因此 dataset 初始化阶段必须先过滤掉非蛋白结构，避免 DNA 文件进入 train/val split 后在 PyRosetta 环境 reset 或 action 解码阶段失败。

## 修改内容

修改：

```text
model/dataset_module/dataset.py
```

新增预处理函数：

- `count_canonical_protein_residues(path)`：读取 PDB `ATOM` 记录，统计唯一 canonical protein residues。
- `is_protein_structure_file(path, min_protein_residues=1)`：判断结构文件是否至少包含指定数量的 canonical protein residues。
- `filter_protein_structure_files(files, min_protein_residues=1, require_non_empty=True)`：批量过滤结构文件，并输出过滤日志。

当前支持的 canonical protein residue names：

```text
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
```

`ProteinStructureDataset.from_folder(...)` 现在会在初始化时完成预处理：

- 新建 train/val index 时：先递归发现 `.pdb/.ent` 文件，再过滤合法蛋白结构，最后划分 train/val。
- 读取已有 train/val index 时：仍会重新检查 index 中的文件；如果发现 DNA/非蛋白文件，会从 index 中移除并重写 index。
- 训练集最终不能为空；验证集允许为空。

修改：

```text
model/dataset_module/__init__.py
```

导出新增的 dataset 预处理函数，方便后续脚本或测试复用。

修改：

```text
training.py
```

新增命令行参数：

```bash
--min-protein-residues
```

默认值为 `1`，表示结构文件中至少包含 1 个 canonical protein residue 才能进入 train/val 候选。该参数会传入 `ProteinStructureDataset.from_folder(...)`，并在 dataset ready 日志中输出。

## 测试更新

修改：

```text
tests/test_dataset.py
```

新增覆盖：

- canonical protein residue 计数；
- DNA-only PDB 被过滤；
- mixed DNA/protein PDB 只要包含 canonical protein residue 就可作为蛋白候选；
- dataset split 不会把 DNA-only 文件写入 train/val index；
- 已存在 index 中的 DNA-only 文件会被过滤并触发 index 重写。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/dataset_module/__init__.py \
  model/dataset_module/dataset.py \
  tests/test_dataset.py
```

dataset 测试：

```text
5 passed in 0.18s
```

全量测试：

```text
86 passed, 1 skipped in 5.15s
```

`mprl-vgpt` 环境下文件夹训练冒烟测试：

```bash
conda run -n mprl-vgpt python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-protein-filter-smoke \
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

- 预处理基于 PDB 文本中的 `ATOM` records 和 canonical residue names，不会启动 PyRosetta。
- DNA/RNA-only 结构会被过滤；蛋白-DNA 复合物只要包含 canonical protein residue，仍会被视为合法蛋白结构候选。
- 只含非标准氨基酸或修饰残基、且没有 canonical residue name 的结构会被过滤；如果后续需要支持 MSE 等修饰残基，可以扩展 residue allowlist 或增加 PyRosetta 级别的结构检查。

# 2026-06-18 缺失 backbone atoms 的结构处理

## 背景

部分用于训练的结构存在局部 atom 缺失，`model/reward_module/reward_calculators.py` 中 `_coordinates_for_local_rmsd(...)` 在计算 local RMSD 时会因为缺少 `N/CA/C/O` 直接抛出异常，例如：

```text
ValueError: Residue 63 (ACY) lacks atom 'N', required for local RMSD
```

该问题会中断长时间训练。此次修改采用两层防护：

1. dataset 初始化阶段过滤 backbone 缺失比例过高的结构；
2. reward 计算阶段对局部缺失 atom 做运行时容错。

## Dataset 侧修改

修改：

```text
model/dataset_module/dataset.py
model/dataset_module/__init__.py
training.py
```

新增：

- `DEFAULT_BACKBONE_ATOMS = ("N", "CA", "C", "O")`
- `DEFAULT_MAX_MISSING_BACKBONE_FRACTION = 0.05`
- `backbone_missing_fraction(path, backbone_atoms=("N", "CA", "C", "O"))`

`is_protein_structure_file(...)` 和 `filter_protein_structure_files(...)` 现在除了检查 canonical protein residues，还会检查 canonical residues 中缺失 backbone atoms 的比例。

训练入口新增参数：

```bash
--max-missing-backbone-fraction
```

默认值为 `0.05`。含义：如果一个结构中超过 5% 的 canonical protein residues 缺失 `N/CA/C/O` 中任意 atom，则该结构不会进入 train/val index。已有 index 在读取时也会被重新检查；如果 index 中包含不合格结构，会被自动重写。

## Reward 侧修改

修改：

```text
model/reward_module/reward_calculators.py
training.py
```

`StepRewardCalculator` 新增参数：

```python
rmsd_missing_atom_policy: str = "penalize"
rmsd_missing_penalty: float = 5.0
min_rmsd_atoms: int = 3
```

支持策略：

- `raise`：保持原始行为，遇到缺失 RMSD atom 直接抛错，适合调试数据。
- `skip_residue`：local RMSD 只使用 reference/mobile 两边都具备完整 RMSD atoms 的 residue；如果剩余 atom 数不足则抛错。
- `penalize`：默认策略。能跳过缺失 residue 时正常计算；如果剩余 atom 不足 `min_rmsd_atoms`，返回 `rmsd_missing_penalty` 作为 local RMSD 惩罚，避免训练中断。

`StepRewardResult` 新增记录字段：

```python
local_rmsd_status
local_rmsd_atom_count
skipped_local_rmsd_residues
```

这些字段会进入 `to_dict()`，从而可以被训练日志记录，用于后续分析哪些结构或 residue 经常触发 RMSD 容错。

训练入口新增参数：

```bash
--rmsd-missing-atom-policy
--rmsd-missing-penalty
--min-rmsd-atoms
```

默认使用：

```bash
--rmsd-missing-atom-policy penalize
--rmsd-missing-penalty 5.0
--min-rmsd-atoms 3
```

## 测试更新

新增/修改：

```text
tests/test_dataset.py
tests/test_reward_calculators.py
```

覆盖：

- PDB 文本中 canonical residue 的 backbone missing fraction 计算；
- backbone 缺失比例过高的结构不会进入 dataset index；
- local RMSD 在 `skip_residue` 策略下跳过缺失 atom 的 residue；
- local RMSD 在 `penalize` 策略下对可用 atom 不足的情况返回惩罚值。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/reward_module/reward_calculators.py \
  model/dataset_module/dataset.py \
  tests/test_reward_calculators.py \
  tests/test_dataset.py
```

聚焦测试：

```text
8 passed in 0.18s
```

全量测试：

```text
89 passed, 1 skipped in 5.17s
```

`mprl-vgpt` 环境下文件夹训练冒烟测试：

```bash
conda run -n mprl-vgpt python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-missing-atoms-smoke \
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

- dataset 侧检查只基于 PDB 文本 `ATOM` records，不运行 PyRosetta 修复或补 atom。
- reward 侧只对 local RMSD 的缺失 atom 做容错；如果 PyRosetta 在 load、mutate、repack 或 minimize 阶段因为结构质量失败，仍需要依赖环境的 `--continue-on-update-error` 或进一步的数据清洗。
- 非标准残基如 `ACY` 不会作为 canonical protein residue 进入 dataset backbone 完整性统计，但如果 PyRosetta pose 的 local neighborhood 包含这类 residue，reward 的 RMSD 容错会跳过或惩罚，而不是默认中断训练。

# 2026-06-18 Environment 侧 PyRosetta 加载前 PDB 清洗

## 背景

训练时在 `model/environment_module/environment.py` 的 `load_pose()` 阶段遇到 PyRosetta 报错：

```text
ERROR: too many tries in fill_missing_atoms!
```

该错误通常发生在 `pose_from_pdb(...)` 内部构建缺失 atom 时。此前 dataset 和 reward 已经做了过滤与 RMSD 容错，但如果结构在 PyRosetta load 阶段就失败，reward 侧无法介入。因此本次在 environment backend 的 `pose_from_pdb(...)` 之前增加 PDB 文本级清洗。

## 修改内容

修改：

```text
model/environment_module/environment.py
training.py
tests/test_environment.py
```

### PyRosettaPoseBackend 清洗逻辑

`PyRosettaPoseBackend.load_pose(...)` 现在默认先调用 PDB cleaner，再调用 `pyrosetta.pose_from_pdb(...)`。

清洗规则：

- 只处理 PDB `ATOM` records；
- 只保留 canonical amino-acid residue names：

```text
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
```

- 丢弃非标准 residue，例如 `ACY`；
- 丢弃缺失 backbone atoms 的 canonical residue，要求同时具备：

```text
N CA C O
```

- 如果 canonical residues 中缺失 backbone 的比例超过阈值，则不再尝试 PyRosetta load，而是提前抛出更清晰的 `ValueError`。

新增数据结构：

```python
PDBCleaningResult
```

记录：

- 原始 PDB 路径；
- 实际传给 PyRosetta 的路径；
- 是否生成 cleaned PDB；
- total / kept / skipped noncanonical / skipped missing backbone residue 数量；
- missing backbone fraction。

### 临时 cleaned PDB

如果原始 PDB 中存在需要剔除的 residue，backend 会生成一个临时 `.cleaned.pdb`：

- 默认写入系统临时目录；
- 默认在 PyRosetta load 结束后删除；
- 可以通过参数保留，便于排查。

### 训练入口参数

`training.py` 新增：

```bash
--no-clean-pdb-before-load
--keep-cleaned-pdbs
--cleaned-pdb-dir
```

默认行为：

- 启用 environment-side PDB 清洗；
- cleaned PDB 临时生成，加载后删除；
- load 阶段的缺失 backbone 阈值复用 dataset 参数 `--max-missing-backbone-fraction`。

如果需要排查某个结构如何被清洗，可以使用：

```bash
python training.py \
  --pdb-dir /path/to/pdb_dir \
  --keep-cleaned-pdbs \
  --cleaned-pdb-dir outputs/cleaned_pdb_debug
```

## 测试更新

修改：

```text
tests/test_environment.py
```

新增覆盖：

- cleaner 会移除非标准 residue 和缺失 backbone 的 canonical residue；
- cleaner 会在缺失 backbone 比例超过阈值时提前报错。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/environment_module/environment.py \
  tests/test_environment.py
```

环境测试：

```text
13 passed in 0.21s
```

全量测试：

```text
91 passed, 1 skipped in 5.15s
```

`mprl-vgpt` 环境下文件夹训练冒烟测试：

```bash
conda run -n mprl-vgpt python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --episodes 1 \
  --max-steps 1 \
  --output-dir /tmp/mprl-env-clean-smoke \
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

- 该 cleaner 是“删除坏 residue”，不是补全 atom；如果你希望保留所有 residue 并重建缺失 atoms，需要单独引入结构修复流程。
- 只清洗 `ATOM` records；ligand、DNA/RNA、water、ion 等 `HETATM` 默认不会进入 cleaned PDB。
- 如果某个结构被清洗后 pose residue numbering 改变，固定传入的 `--mutable-positions` 需要谨慎使用；默认 mutable positions 会基于清洗后的 Pose 自动生成。

# 2026-06-18 数据集 epoch/batch 训练调度

## 背景

此前 `training.py` 只通过 `--episodes` 指定一个固定 episode 总数，然后每个 episode 从训练集随机采样一个 PDB。这种方式可以运行，但语义更像“固定步数采样”，不太像标准机器学习/强化学习项目中对一个数据集进行多轮 epoch 训练。

本次修改在保留旧 `--episodes` 兼容模式的同时，新增 dataset-style 多 batch 训练：

- 一个 epoch 表示对训练集 PDB 的一轮 episode 调度；
- 一个 dataset batch 表示一组 PDB episode；
- batch 内每个 PDB 仍然独立 reset 环境并运行一个 RL episode；
- replay buffer 和 DDQN optimize 逻辑保持不变，仍然在 step 级别累积 transition 和优化。

## Dataset 侧修改

修改：

```text
model/dataset_module/dataset.py
```

新增：

```python
ProteinStructureDataset.train_epoch_paths(...)
ProteinStructureDataset.iter_train_batches(...)
```

行为：

- `train_epoch_paths(...)` 返回一个 epoch 内要训练的 PDB 路径序列；
- 默认 `episodes_per_epoch=None` 时，一个 epoch 覆盖完整训练集；
- 如果 `episodes_per_epoch` 大于训练集大小，会重复 shuffled cycles；
- `iter_train_batches(...)` 按 `batch_size` 将 epoch paths 切分为 PDB batch。

## Training 侧修改

修改：

```text
training.py
```

新增训练参数：

```bash
--epochs
--episodes-per-epoch
--train-batch-size
--no-shuffle-train
```

推荐使用方式：

```bash
python training.py \
  --pdb-dir /path/to/pdb_dir \
  --epochs 10 \
  --train-batch-size 8
```

含义：

- `--epochs 10`：对训练集进行 10 轮 episode 调度；
- `--episodes-per-epoch`：每个 epoch 训练多少个 PDB episode；默认等于训练集大小；
- `--train-batch-size 8`：每个 dataset batch 包含 8 个 PDB episode；
- `--no-shuffle-train`：关闭每个 epoch 内的 PDB 顺序打乱。

兼容性：

- 如果没有传入 `--epochs`，训练仍使用旧的 `--episodes` 模式；
- `--episodes` 现在标记为 legacy total episode count；
- candidate、validation、checkpoint 仍然使用全局 episode index 触发，不破坏已有日志和输出结构。

日志增强：

每个 episode 的日志和 `logger.end_episode(..., extra=...)` 现在包含：

```text
epoch
batch_index
batch_item_index
planned_episodes
training_schedule
```

方便后续按 epoch/batch 分析 reward、epsilon、optimization steps 和 candidate PDB。

## 测试更新

修改：

```text
tests/test_dataset.py
```

新增覆盖：

- `iter_train_batches(...)` 在一个 epoch 中覆盖完整训练集；
- `train_epoch_paths(...)` 在 `episodes_per_epoch` 大于训练集大小时会重复路径序列。

## 验证结果

编译检查：

```bash
python -m py_compile \
  training.py \
  model/dataset_module/dataset.py \
  tests/test_dataset.py
```

dataset 聚焦测试：

```text
8 passed in 0.19s
```

全量测试：

```text
93 passed, 1 skipped in 5.06s
```

`mprl-vgpt` 环境下 epoch/batch 训练冒烟测试：

```bash
conda run -n mprl-vgpt python training.py \
  --pdb-dir model/reward_module \
  --mode single \
  --device cpu \
  --epochs 1 \
  --episodes-per-epoch 1 \
  --train-batch-size 2 \
  --max-steps 1 \
  --output-dir /tmp/mprl-epoch-batch-smoke \
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

- 当前 `train-batch-size` 是“PDB episode batch”的调度单位，不是并行环境 batch；batch 内 PDB 仍然逐个 episode 执行。
- DDQN 的梯度 batch 仍由 replay buffer 的 `micro_batch_size * gradient_accumulation_steps` 控制。
- 如果要进一步提高吞吐量，下一步可以引入 vectorized environments 或多进程 rollout workers。

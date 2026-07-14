# Stdout Logging 增强说明

本次修改目标是解决长时间训练期间 stdout 没有持续反馈的问题。实现方式统一改为 Python 标准库 `logging`，由 `training.py` 配置 root logger 输出到 stdout，`training.py` import 的项目模块使用各自的 module logger 继续向 stdout 传播日志。

## 使用方式

默认训练会以 `INFO` 级别输出大量进度信息：

```bash
python training.py --mode single --device cuda:2
```

需要更细粒度的 micro-batch 等内部信息时：

```bash
python training.py --mode single --device cuda:2 --log-level DEBUG
```

如果觉得 step 级进度输出太频繁，可以调大：

```bash
python training.py --mode single --device cuda:2 --log-every-steps 10
```

日志格式：

```text
YYYY-MM-DD HH:MM:SS | LEVEL | logger.name | message
```

## training.py 修改

- 新增 `logging`、`sys`、`time`，并通过 `configure_stdout_logging()` 将日志输出到 stdout。
- 新增 CLI 参数：
  - `--log-level`：控制 stdout 日志级别，默认 `INFO`。
  - `--log-every-steps`：控制训练循环额外 progress 日志频率，默认每个环境 step 输出一次。
- 启动时输出：
  - 完整 args；
  - Torch 版本；
  - CUDA build 版本；
  - CUDA 是否可用；
  - GPU 数量和每张 GPU 名称。
- 训练模式输出：
  - single/multi 模式；
  - 使用的 device；
  - multi 模式的 gpu ids；
  - 是否启用 DataParallel。
- 环境、agent、replay buffer、logger 构建阶段输出配置摘要。
- 每个 episode 输出：
  - episode 开始；
  - reset 后 valid action 数；
  - episode 总 reward、step 数、epsilon、优化步数和耗时。
- 每个 environment step 输出：
  - 当前 episode/step/global step；
  - epsilon；
  - valid action 数；
  - 选中的 action；
  - decoded action；
  - reward、step reward、terminal reward；
  - accepted/reason；
  - truncated/terminated；
  - 下一步 valid action 数；
  - step 耗时。
- ReplayBuffer add 后输出当前 size/capacity/position。
- optimize 时输出：
  - 是否因 warmup 未满足而跳过；
  - 若优化成功，输出 loss、mean Q、target Q、TD error、grad norm、target sync 状态。
- 保存产物时输出：
  - candidate PDB 路径；
  - periodic checkpoint 路径；
  - final checkpoint 路径；
  - final plots 数量和路径。

## model/environment_module/environment.py 修改

- 新增 module logger。
- `PyRosettaPoseBackend` 增加日志：
  - PyRosetta import；
  - `pyrosetta.init()` 是否调用；
  - score function 构建；
  - PDB 加载路径、残基数、耗时；
  - local residues 计算中心、半径、数量、耗时；
  - mutate 开始/结束；
  - local repack 开始/结束和耗时；
  - local minimization 开始/结束和耗时；
  - dump PDB 路径。
- `MechanicalProteinEnv` 增加日志：
  - 初始化输入和最终环境尺寸；
  - residue 数、mutable position 数、action dim、observation shape；
  - reset 开始/完成、valid actions、sequence、mutable sequence；
  - step 开始，包含 action、pose position、target amino acid；
  - invalid action 的原因和 penalty；
  - candidate pose clone；
  - step reward 原始值、缩放值、metrics；
  - update error 的 exception 和 penalty；
  - accepted mutation 和 accepted mutation 计数；
  - truncation reason；
  - terminal reward 是否计算、原始值、缩放值、metrics；
  - step 完成摘要和耗时；
  - current pose 保存路径。

## model/agent_module/ddqn_agent.py 修改

- 新增 module logger。
- agent 初始化时输出：
  - state shape；
  - action dim；
  - DDQNConfig；
  - resolved device；
  - AMP 是否启用；
  - effective batch size；
  - 初始化耗时。
- action selection 输出：
  - action；
  - random/greedy；
  - evaluate 模式；
  - epsilon；
  - valid action 数；
  - environment step 计数。
- replay warmup 未满足时输出 buffer size 和 required size。
- replay batch sample 前输出 batch size 和 buffer size。
- optimize batch 输出：
  - total size；
  - micro batch size；
  - device；
  - AMP；
  - optimizer step grad norm；
  - target sync；
  - loss、Q value、target Q、TD error、micro batch 数和耗时。
- `DEBUG` 级别额外输出每个 micro-batch 的 start/stop、loss sum 和耗时。
- checkpoint 保存和 hard target sync 增加开始/完成日志。

## model/replay_buffer_module/replay_buffer.py 修改

- 新增 module logger。
- 初始化输出：
  - capacity；
  - state shape；
  - action dim；
  - 是否保存 action masks；
  - seed。
- 每次 `add()` 输出：
  - 写入 index；
  - action；
  - reward；
  - terminated/truncated/done；
  - 当前 size；
  - next position。
- 每次 `sample()` 输出：
  - batch size；
  - replace；
  - buffer size；
  - 前 10 个采样 index 预览。
- `save()` 和 `load()` 输出路径、size 和 capacity。

## model/logging_module/training_logger.py 修改

- 新增 module logger。
- 初始化输出：
  - TrainingLoggerConfig；
  - output/log/plot/tensorboard 目录；
  - resume 时已有 episode/optimization/step 记录数。
- 每次写记录输出：
  - optimization record 的 optimization step、global step、loss、JSONL 路径；
  - step record 的 episode、episode step、global step、reward、done、JSONL 路径；
  - episode record 的 episode、total reward、steps、JSONL 和 CSV 路径。
- JSONL append 输出路径和字段名。
- JSONL resume 读取输出路径和记录数；不存在时说明从空记录开始。
- episode CSV 写入输出路径和行数。
- plot 生成输出：
  - 输入记录数量；
  - plots 输出目录；
  - 最终生成图的数量和路径。
- TensorBoard writer 创建、flush、close 输出日志。

## 注意事项

- 默认 `INFO` 现在会非常密集，适合调试长测试是否卡住。
- 如果后续正式长跑希望减少 stdout 压力，可以使用 `--log-every-steps` 放大训练入口 progress 间隔，并将部分模块日志级别下调为 `DEBUG`。
- 本次修改不改变训练逻辑、reward 逻辑、模型结构、ReplayBuffer 数据结构或 checkpoint 格式。

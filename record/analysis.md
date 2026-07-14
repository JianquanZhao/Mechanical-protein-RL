# Mechanical-protein-RL 代码分析

## 1. 项目定位

当前目录是一个用于增强力学蛋白力学性能的强化学习初始版本。整体思路是用离散动作表示氨基酸突变，环境负责在 PyRosetta 中执行候选突变、局部重打包和局部最小化，奖励模块把结构稳定性指标和终端力学预测目标转成标量 reward，DDQN agent 通过 replay buffer 进行 masked Double DQN 训练。

代码已经按模块拆分为：

- `model/environment_module`: 蛋白突变环境，Gymnasium 风格接口。
- `model/reward_module`: step reward 和 terminal reward 计算器。
- `model/replay_buffer_module`: DDQN 均匀经验回放。
- `model/agent_module`: masked Double DQN agent。
- `model/logging_module`: 训练日志、CSV/JSONL、图像和可选 TensorBoard。
- `tests`: 单元测试和一个可选 PyRosetta 集成 smoke test。

## 2. 运行链路

标准训练链路可以概括为：

1. `MechanicalProteinEnv.reset()` 从 wild type/reference pose 开始，返回 observation 和 `info["action_mask"]`。
2. `DDQNAgent.select_action(state, action_mask=...)` 使用 epsilon-greedy 策略，从合法突变动作中选一个 action。
3. `MechanicalProteinEnv.step(action)` 解码 action，克隆当前 pose，执行突变、局部 repack、局部 minimize，再调用 `StepRewardCalculator.evaluate(...)`。
4. 环境返回 `next_state, reward, terminated, truncated, next_info`。本项目基础版本没有自然 biological terminal condition，通常通过 `max_steps` 或动作耗尽产生 `truncated=True`。
5. `ReplayBuffer.add(...)` 存储 transition，包括当前和下一状态 action mask。
6. `DDQNAgent.optimize_from_replay_buffer(...)` 达到 warmup 后采样 effective batch，做 Double DQN 更新。
7. `TrainingLogger` 记录 step、episode 和 optimization 指标，并生成诊断图。

## 3. 环境模块

核心文件：`model/environment_module/environment.py`

### 3.1 动作空间

动作空间是 `L * 20`，其中 `L` 是可突变位点数量，20 是标准氨基酸数量。

动作解码规则：

```text
mutable_position_index = action // n_amino_acids
amino_acid_index = action % n_amino_acids
```

`mutable_positions` 使用 PyRosetta Pose 的 1-indexed residue position。若用户不传入，则默认选择 reference pose 中所有 canonical amino acid residue。

### 3.2 observation

默认 observation 是可突变位点序列的 one-hot 编码：

- `flatten_observation=True`: shape 为 `(L * 20,)`。
- `flatten_observation=False`: shape 为 `(L, 20)`。

也支持传入 `observation_encoder(pose, env)`，用于后续加入结构描述符、embedding 或力学相关特征。

### 3.3 action mask

`action_mask()` 返回 shape 为 `(L * 20,)` 的 bool 数组：

- 屏蔽 no-op，即突变成当前氨基酸。
- 若 `prevent_revisit_positions=True`，已经突变过的位置会整块屏蔽。

这与 DDQN agent 的 masked action selection 和 masked target calculation 是对齐的。

### 3.4 PyRosetta 后端

`PyRosettaPoseBackend` 懒加载 PyRosetta，因此 import 环境模块本身不要求安装 PyRosetta。默认后端完成：

- `pose_from_pdb` 加载初始结构。
- `MutateResidue` 执行单点突变。
- `PackRotamersMover` 对局部邻域做 side-chain repack。
- `MinMover` 对局部 chi torsion 做最小化，可选局部 backbone。
- `dump_pdb` 保存当前候选结构。

同时定义了 `PoseBackend` protocol，测试中用 fake backend 替代 PyRosetta，这让环境逻辑可以纯 Python 测试。

### 3.5 episode 结束

当前实现中 `terminated` 总是 False，episode 结束通过 `truncated` 表示：

- 达到 `max_steps`。
- 若 `truncate_when_no_valid_actions=True`，合法动作耗尽。

episode 结束时会调用可选 `TerminalRewardCalculator.evaluate_pose()`，并把 terminal reward 加到最后一步 reward 中。

## 4. 奖励模块

核心文件：`model/reward_module/reward_calculators.py`

### 4.1 StepRewardCalculator

`StepRewardCalculator` 只评估 pose，不负责修改 pose。它需要 PyRosetta，使用当前 pose、previous pose 和 reference pose 计算结构层面的 step reward。

当前 step reward 由四类指标组成：

- collision：基于 Rosetta `fa_rep`，可选加 `fa_intra_rep`，通常作为惩罚项。
- backbone hydrogen bond delta：backbone-backbone H-bond 数量变化，作为奖励项。
- sidechain-involving hydrogen bond delta：涉及侧链的 H-bond 数量变化，作为奖励项。
- local RMSD：局部 Kabsch superposition 后 RMSD，作为结构漂移惩罚。

公式结构是：

```text
reward =
  - collision_weight * normalized(collision_loss)
  + backbone_hbond_weight * normalized(backbone_delta)
  + sidechain_hbond_weight * normalized(sidechain_delta)
  - local_rmsd_weight * normalized(local_rmsd)
```

返回的 `StepRewardResult` 保留了原始指标、delta、局部 residue、reward components，适合训练诊断和后续 reward hacking 检查。

### 4.2 TerminalRewardCalculator

`TerminalRewardCalculator` 设计目标是封装一个力学性能预测器，输出：

- `max_stress`
- `toughness`

支持 sklearn-like `.predict()`、callable、mapping 输出、z-score 标准化、absolute/delta 两种 reward mode。

重要缺口：当前 `_predict_one()` 中真实 predictor 调用被注释掉了：

```python
prediction = np.array([0., 0.]) # self._run_predictor(batch)
```

因此 terminal reward 目前固定为 0，终端力学目标还没有真正接入。这是后续“颗粒度对齐”里最优先需要修复或确认的地方之一。

## 5. Replay Buffer 模块

核心文件：`model/replay_buffer_module/replay_buffer.py`

`ReplayBuffer` 是固定容量、预分配 NumPy 数组的均匀经验回放。它存储：

- `states`
- `actions`
- `rewards`
- `next_states`
- `terminateds`
- `truncateds`
- `dones`
- `action_masks`
- `next_action_masks`

`done` 默认由 `terminated or truncated` 推断；若显式传入且不一致会报错。采样返回 `ReplayBatch`，并对数组做 copy，避免外部修改 buffer 内部数据。

它支持：

- ring buffer 覆盖旧 transition。
- 可复现实验的随机种子。
- 保存/加载 `.npz` snapshot。
- 可关闭 action mask 存储，但当前环境和 DDQN 训练建议保持开启。

## 6. DDQN Agent 模块

核心文件：`model/agent_module/ddqn_agent.py`

### 6.1 网络结构

默认 `QNetwork` 是简单 MLP：

- 输入：flatten 后的 observation。
- 输出：每个离散动作的 Q value。
- hidden dims 默认 `(256, 256)`。

也可以传入自定义 `online_network` 和 `target_network`，只要输出 shape 为 `(batch, action_dim)`。

### 6.2 探索和动作选择

`DDQNAgent.select_action(...)` 支持：

- epsilon-greedy exploration。
- evaluation mode 下纯 greedy。
- action mask，非法动作 Q value 会设为 `-inf`。
- 若当前 mask 没有合法动作，会直接报错。

epsilon 是线性衰减：

```text
epsilon_start -> epsilon_end over epsilon_decay_steps
```

### 6.3 Double DQN target

target 计算为：

```text
a* = argmax_a Q_online(next_state, a)
target = reward + gamma * (1 - done) * Q_target(next_state, a*)
```

实现中会先对 `next_action_masks` 做 masking。terminal row 允许 all-False mask，因为 bootstrap 项为 0；非 terminal row 若 all-False，会报错，提示环境动作终止逻辑不一致。

### 6.4 梯度累积

`DDQNConfig` 中：

```text
effective_batch_size = micro_batch_size * gradient_accumulation_steps
```

`optimize_batch()` 会按 `micro_batch_size` 切分 batch，每个 micro-batch 的 Huber loss 用 sum reduction，再除以 total effective batch size，因此累积梯度等价于完整 batch mean loss 的梯度。

它还支持：

- Smooth L1 / Huber loss。
- gradient clipping。
- hard target sync。
- soft target sync。
- CUDA AMP，可选。
- checkpoint 保存/恢复，包括网络、optimizer、epsilon 进度、NumPy/Torch RNG。

## 7. Logging 模块

核心文件：`model/logging_module/training_logger.py`

`TrainingLogger` 与 PyRosetta 和 DDQN 实现解耦，只接受 dict、dataclass 或带 `to_dict()` 的对象。输出目录结构为：

```text
outputs/ddqn_base/
├── logs/
│   ├── episodes.jsonl
│   ├── episodes.csv
│   ├── optimization.jsonl
│   └── steps.jsonl
├── plots/
└── tensorboard/
```

它能记录：

- optimizer 指标：loss、TD error、grad norm、Q values、epsilon 等。
- step 指标：reward、step reward components、terminal reward、sequence。
- episode 指标：total reward、episode length、epsilon、optimization steps、sequence。

并生成诊断图：

- episode reward
- episode length
- epsilon
- optimization loss
- TD error
- grad norm
- Q values
- step reward
- reward components
- terminal reward

## 8. 示例脚本和文档

主要文档和示例：

- `README_DDQN_AGENT.md`: DDQN agent 用法、梯度累积、target 公式、checkpoint。
- `model/environment_module/README_ENVIRONMENT.md`: 环境用法、动作空间、PyRosetta 集成。
- `model/replay_buffer_module/README_REPLAY_BUFFER.md`: replay buffer 用法。
- `README_TRAINING_LOGGER.md`: logger 集成方式。
- `model/agent_module/example_training.py`: 环境、buffer、agent 的端到端训练示例。
- `model/logging_module/example_training_with_logging.py`: 带 logger 的训练示例。

顶层 `README.md` 目前只有项目名，尚未整合完整入口说明。

## 9. 测试现状

已运行：

```bash
pytest -q
```

结果：

```text
72 passed, 1 skipped in 35.02s
```

跳过的是可选真实 PyRosetta smoke test，条件是环境中没有安装 PyRosetta 或缺少对应运行条件。其余单元测试覆盖：

- 环境 action decode、mask、step、rollback、terminal reward、history 保存。
- DDQN epsilon、masked selection、Double DQN target、terminal mask、梯度累积、target sync、checkpoint。
- ReplayBuffer add/sample/save/load、mask、ring overwrite、输入校验。
- TrainingLogger JSONL/CSV/TensorBoard/plot/resume 行为。

补充验证：使用 `conda activate mprl` 后，`mprl` 环境中的 PyRosetta 可以正常 import，且真实 PyRosetta 环境 smoke test 已通过：

```bash
pytest -q tests/test_environment_pyrosetta_smoke.py
```

结果：

```text
1 passed in 8.69s
```

完整 `pytest -q` 在 `mprl` 环境下当前无法完成 collection，因为该环境是 Python 3.8.20，而 `tests/test_ddqn_agent.py` 中使用了 Python 3.10+ 的类型注解语法 `np.ndarray | None`。如果希望在 `mprl` 中跑完整测试，需要把该注解改成 Python 3.8 兼容写法，例如 `Optional[np.ndarray]`，或把 `mprl` 升级到 Python 3.10+。

## 10. 当前成熟度判断

这是一个结构清晰、接口边界比较干净的初始版本。模块之间已经基本对齐：

- 环境输出 `action_mask`。
- buffer 保存 `next_action_mask`。
- agent 用 `next_action_mask` 做 DDQN target masking。
- logger 能抽取 reward components 和 terminal reward。
- tests 避免强依赖 PyRosetta，便于快速迭代。

但它还更像“可训练骨架”和“接口原型”，不是已经完成科学闭环的版本。主要原因是 terminal mechanics predictor 尚未真正接入，reward 标定和 observation 粒度仍然偏基础。

## 11. 主要风险和缺口

1. Terminal reward 未真实工作  
   `TerminalRewardCalculator._predict_one()` 固定返回 `[0., 0.]`，导致 max stress 和 toughness 不会影响训练。

2. observation 目前默认只有序列 one-hot  
   若目标是力学性能增强，可能需要加入结构局部环境、能量项、二级结构、接触图、embedding 或候选区域特征，否则 agent 很难学习结构和力学之间的关系。

3. step reward 权重和 scale 需要数据标定  
   `StepRewardWeights` 和 `StepRewardScales` 默认值都是基础占位。实际训练前应基于 pilot mutation set 统计各项指标分布，否则某个 reward component 可能主导训练。

4. 顶层训练入口还偏示例化  
   `example_training.py` 可跑通链路，但还没有形成正式 CLI/config 驱动的实验入口，也缺少统一的 checkpoint resume、实验配置保存和随机种子管理策略。

5. PyRosetta 真实路径只 smoke test  
   真实结构更新和 reward 计算成本较高，目前只有可选 smoke test。后续需要针对真实 PDB、小 mutable region、固定随机种子建立更稳定的集成测试。

6. 多目标优化目前被标量化  
   terminal reward 支持 `objective_vector`，但 DDQN 训练仍是 scalar reward。若 max stress 和 toughness 存在 trade-off，后续可能需要 Pareto selection、constraint reward 或多目标日志分析。

## 12. 后续颗粒度对齐建议

建议下一阶段按以下顺序对齐：

1. 明确科学任务颗粒度  
   确认目标蛋白、mutable region、每个 episode 的最大突变数、是否允许同一位置重复突变、是否允许 backbone 最小化。

2. 接入真实 terminal predictor  
   恢复 `_run_predictor(batch)` 调用，确定 `feature_extractor` 输入格式、预测输出单位、normalization mean/std、wild-type baseline。

3. 标定 step reward  
   对 wild type 附近随机突变做 pilot sampling，统计 collision、H-bond delta、local RMSD 的范围，再设置 `StepRewardScales` 和权重。

4. 升级 observation  
   从 sequence one-hot 扩展到包含结构和局部力学相关信息的 state representation，并同步调整 `state_shape`、agent 网络结构和测试。

5. 固化训练入口  
   把示例脚本整理成正式训练脚本，支持配置文件、resume、输出目录、checkpoint 周期、logger、最终 candidate 保存。

6. 建立真实 PyRosetta 小规模验收  
   用一个小 mutable region 和短 episode 作为 CI/本地验收流程，检查 reward 是否有限、action mask 是否合理、候选 PDB 是否可保存、日志是否完整。

## 13. 我对当前代码的心智模型

当前项目可以看成五层：

```text
PyRosetta structure update
        ↓
MechanicalProteinEnv: mutation MDP + action mask
        ↓
ReplayBuffer: transition storage with masks
        ↓
DDQNAgent: masked Double DQN optimization
        ↓
TrainingLogger: experiment observability
```

真正需要补齐的科学闭环是：

```text
candidate sequence/structure
        ↓
mechanical predictor / simulation-derived objective
        ↓
terminal reward and calibrated step reward
        ↓
policy improvement
```

也就是说，工程框架已经搭起来了；下一步的关键不是再加很多训练技巧，而是把“力学性能目标、结构状态表示、reward 标定、真实 PyRosetta 运行成本”这几件事对齐到同一个实验定义里。

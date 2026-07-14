# Training Logging and Visualization

## Suggested project structure

Copy `logging_module` into the existing project:

```text
.
├── model
│   ├── agent_module
│   │   └── ddqn_agent.py
│   ├── environment_module
│   │   └── environment.py
│   ├── logging_module
│   │   ├── __init__.py
│   │   ├── training_logger.py
│   │   └── example_training_with_logging.py
│   ├── replay_buffer_module
│   │   └── replay_buffer.py
│   └── reward_module
│       ├── reward_calculators.py
│       └── wild_type.pdb
└── tests
    └── test_training_logger.py
```

## Install dependencies

```bash
python -m pip install numpy matplotlib pytest
```

TensorBoard is optional:

```bash
python -m pip install tensorboard
```

## Run tests

```bash
python -m pytest -q tests/test_training_logger.py
```

## Generated outputs

```text
outputs/ddqn_base/
├── logs/
│   ├── episodes.jsonl
│   ├── episodes.csv
│   ├── optimization.jsonl
│   └── steps.jsonl
├── plots/
│   ├── episode_reward.png
│   ├── episode_length.png
│   ├── epsilon.png
│   ├── optimization_loss.png
│   ├── td_error.png
│   ├── grad_norm.png
│   ├── q_values.png
│   ├── step_reward.png
│   ├── reward_components.png
│   └── terminal_reward.png
└── tensorboard/
```

## Minimal integration

```python
from model.logging_module.training_logger import (
    TrainingLogger,
    TrainingLoggerConfig,
)

logger = TrainingLogger(
    TrainingLoggerConfig(
        output_dir="outputs/ddqn_base",
        rolling_window=20,
        plot_every_episodes=10,
        gradient_clip_threshold=10.0,
    )
)
```

After every optimizer update:

```python
if optimization_result is not None:
    logger.log_optimization(
        optimization_result,
        global_step=total_environment_steps,
    )
```

After every environment transition:

```python
logger.log_step(
    episode=episode,
    episode_step=episode_steps,
    global_step=total_environment_steps,
    reward=reward,
    terminated=terminated,
    truncated=truncated,
    info=next_info,
)
```

After every episode:

```python
logger.end_episode(
    episode=episode,
    total_reward=episode_reward,
    episode_steps=episode_steps,
    epsilon=agent.epsilon,
    optimization_steps=agent.optimization_steps,
    info=info,
)
```

Before the program exits:

```python
logger.generate_plots()
logger.close()
```

## Optional TensorBoard

Enable TensorBoard event files:

```python
TrainingLoggerConfig(
    enable_tensorboard=True,
)
```

Run the viewer:

```bash
tensorboard --logdir outputs/ddqn_base/tensorboard
```

## First curves to inspect

- `episode_reward.png`: whether reward improves and whether variance is extreme.
- `optimization_loss.png`: whether Huber loss is decreasing or diverging.
- `td_error.png`: whether Bellman errors become smaller.
- `grad_norm.png`: whether clipping is triggered frequently.
- `q_values.png`: whether predicted Q values drift far above target Q values.
- `reward_components.png`: whether one reward term dominates the others.

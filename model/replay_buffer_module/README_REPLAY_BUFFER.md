# DDQN Uniform Replay Buffer

## Project structure

```text
.
├── model
│   ├── environment_module
│   │   └── environment.py
│   ├── replay_buffer_module
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   └── example_usage.py
│   └── reward_module
│       └── reward_calculators.py
└── tests
    └── test_replay_buffer.py
```

## Run tests

```bash
python -m pip install numpy pytest
python -m pytest -q tests/test_replay_buffer.py
```

## Construct the buffer

```python
from model.replay_buffer_module.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(
    capacity=100_000,
    state_shape=observation.shape,
    action_dim=env.action_space.n,
    seed=7,
)
```

## Add one transition

```python
buffer.add(
    state=state,
    action=action,
    reward=reward,
    next_state=next_state,
    terminated=terminated,
    truncated=truncated,
    action_mask=info["action_mask"],
    next_action_mask=next_info["action_mask"],
)
```

## Sample a mini-batch

```python
batch = buffer.sample(batch_size=64)

states = batch.states
actions = batch.actions
rewards = batch.rewards
next_states = batch.next_states
dones = batch.dones
next_action_masks = batch.next_action_masks
```

## DDQN target calculation

The online network selects the next action. The target network evaluates it.
Mask invalid actions before `argmax`.

```python
next_q_online = online_network(next_states)
next_q_online[~next_action_masks] = -inf
next_actions = argmax(next_q_online, dim=1)

next_q_target = target_network(next_states)
target_q = rewards + gamma * (1 - dones) * gather(next_q_target, next_actions)
```

Start with uniform replay. Add prioritized experience replay only after the
end-to-end pipeline is stable.

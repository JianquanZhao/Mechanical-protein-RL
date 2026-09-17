# DDQN Replay Buffer

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

Uniform replay remains the default. The training entry point can additionally
combine TD-error PER with a positive-terminal stratum:

```python
buffer = ReplayBuffer(
    capacity=100_000,
    state_shape=observation.shape,
    action_dim=env.action_space.n,
    sampling_strategy="prioritized",
    positive_sample_fraction=0.25,
    positive_replay_reserve_fraction=0.10,
    # Internal API keeps this compatibility name; CLI uses
    # --positive-reward-lower-bound.
    positive_reward_threshold=0.0,
)
```

A terminal episode enters the positive stratum only when its propagated
`terminal_reward_lcb` is finite and greater than
`positive_reward_threshold`. If no empirical LCB is supplied, the point
terminal reward is used as a fallback, so the threshold acts as a minimum
effect/noise boundary. With n-step replay this includes terminal-bearing
prefixes, not ordinary transitions with a positive local shaping reward.

`positive_sample_fraction` sets the target positive quota in each batch. The
sampler first collapses terminal-bearing n-step rows to one representative per
known `episode_id`, then applies PER by absolute TD error within the positive
and remaining strata. This prevents one successful episode from filling the
positive quota several times. Importance weights correct the within-stratum
PER bias while preserving the intentional positive quota.

`positive_replay_reserve_fraction` changes eviction only after the buffer is
full: once enough positive rows have arrived, negative insertions cannot reduce
their retained transition fraction below the configured reserve. No ESM state
is duplicated, so this policy does not increase replay capacity or state-memory
allocation.

The `ReplayBuffer` API keeps zero as its conservative standalone default; the
training CLI defaults `--positive-sample-fraction` to `0.25`. Useful diagnostics
include `positive_count`, `positive_fraction`,
`positive_unique_episode_count`, and `positive_episode_duplicate_fraction`.

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
    terminal_reward=terminal_reward,
    terminal_reward_lcb=terminal_reward_lcb,
    episode_id=episode_id,
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

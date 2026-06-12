# DDQN Agent for Mechanical-Protein Optimization

## Project structure

Copy `agent_module` into the existing project:

```text
.
├── model
│   ├── agent_module
│   │   ├── __init__.py
│   │   ├── ddqn_agent.py
│   │   └── example_training.py
│   ├── environment_module
│   │   └── environment.py
│   ├── replay_buffer_module
│   │   └── replay_buffer.py
│   └── reward_module
│       ├── reward_calculators.py
│       └── wild_type.pdb
└── tests
    └── test_ddqn_agent.py
```

## Install lightweight dependencies

```bash
python -m pip install numpy torch pytest
```

## Run tests

```bash
python -m pytest -q tests/test_ddqn_agent.py
```

## Gradient accumulation

Set:

```python
DDQNConfig(
    micro_batch_size=16,
    gradient_accumulation_steps=4,
)
```

The effective batch size is:

```text
16 * 4 = 64 transitions
```

Only one micro-batch of 16 transitions is placed on the GPU at a time. Four
backward passes accumulate gradients. The optimizer performs one parameter
update after all four micro-batches.

The code divides every micro-batch loss by the total effective-batch size.
Therefore, the accumulated gradient equals the gradient of the mean loss over
the full effective batch, including when the final micro-batch is smaller.

## DDQN target

The target is:

```text
a* = argmax_a Q_online(next_state, a)
target = reward + gamma * (1 - done) * Q_target(next_state, a*)
```

Invalid actions are masked before `argmax`.

## Terminal masks

A terminal next state may contain an all-False `next_action_mask`. Its bootstrap
term is zero. A non-terminal next state with an all-False mask raises an error,
because it indicates inconsistent environment logic.

## Optional AMP

After the base pipeline is stable on CUDA, memory use and throughput may improve
further with:

```python
DDQNConfig(
    use_amp=True,
)
```

AMP is automatically disabled when CUDA is unavailable.

## Checkpointing

```python
agent.save_checkpoint("checkpoints/agent.pt")

restored = DDQNAgent.from_checkpoint(
    "checkpoints/agent.pt",
    map_location="cpu",
)
```

The checkpoint includes:

- online-network weights;
- target-network weights;
- optimizer state;
- epsilon-schedule progress;
- optimizer-step count;
- NumPy RNG state;
- PyTorch RNG state.

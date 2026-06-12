# Mechanical protein RL environment

## Project layout

```text
.
├── model
│   ├── __init__.py
│   ├── environment_module
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   └── example_environment_usage.py
│   └── reward_module
│       ├── __init__.py
│       ├── reward_calculators.py
│       └── wild_type.pdb
└── tests
    └── test_environment.py
```

Copy `environment.py` and the package `__init__.py` files into your existing
project. Keep your existing `wild_type.pdb` under `model/reward_module`.

## Action space

By default, the action space is `L * 20`, where `L` is the number of mutable
positions. Action decoding is:

```python
mutable_position_index = action // 20
amino_acid_index = action % 20
```

The environment supports a restricted mechanical-lock region:

```python
env = MechanicalProteinEnv(
    initial_pdb_path="model/reward_module/wild_type.pdb",
    mutable_positions=[38, 39, 40, 41, 42, 67, 68, 69, 70],
)
```

## Base DDQN usage

```python
observation, info = env.reset(seed=7)

while True:
    action_mask = info["action_mask"]
    action = choose_action_with_mask(observation, action_mask)
    next_observation, reward, terminated, truncated, info = env.step(action)

    replay_buffer.add(
        observation,
        action,
        reward,
        next_observation,
        terminated or truncated,
        info["action_mask"],
    )

    observation = next_observation
    if terminated or truncated:
        break
```

## Recommended starting configuration

Start conservatively:

```python
env = MechanicalProteinEnv(
    initial_pdb_path="model/reward_module/wild_type.pdb",
    max_steps=5,
    mutable_positions=mechanical_lock_positions,
    local_repack_radius=8.0,
    perform_repack=True,
    perform_minimize=True,
    minimize_backbone=False,
    prevent_revisit_positions=False,
    raise_on_update_error=True,
)
```

During long training runs, change `raise_on_update_error=False` so that a rare
packing/minimization failure rolls back the candidate Pose and produces
`update_error_penalty` instead of interrupting training.

## Run the pure-Python tests

The tests use a fake backend and do not require PyRosetta:

```bash
python -m pip install numpy pytest
python -m pytest -q tests/test_environment.py
```

To run the optional real-PyRosetta smoke test after copying your existing PDB:

```bash
python -m pytest -q tests/test_environment_pyrosetta_smoke.py
```

## PyRosetta integration

`PyRosettaPoseBackend` uses:

1. `pose_from_pdb()` to load the wild-type structure;
2. `MutateResidue` to apply the selected substitution;
3. `standard_packer_task()` with `restrict_to_repacking()` and
   `PackRotamersMover` to repack only the local neighborhood;
4. `MoveMap` and `MinMover` to minimize local chi torsions and, optionally,
   local backbone torsions;
5. your existing `StepRewardCalculator.evaluate(...)` after a candidate update;
6. your optional `TerminalRewardCalculator.evaluate_pose(...)` when the episode
   reaches `max_steps` or is finalized manually.

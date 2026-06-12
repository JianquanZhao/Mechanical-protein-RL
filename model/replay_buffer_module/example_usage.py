"""
Minimal integration example for MechanicalProteinEnv.
"""

from model.environment_module.environment import MechanicalProteinEnv
from model.replay_buffer_module.replay_buffer import ReplayBuffer

env = MechanicalProteinEnv(
    initial_pdb_path="model/reward_module/wild_type.pdb",
    max_steps=5,
)

state, info = env.reset(seed=7)

replay_buffer = ReplayBuffer(
    capacity=100_000,
    state_shape=state.shape,
    action_dim=env.action_space.n,
    seed=7,
    store_action_masks=True,
)

while True:
    action_mask = info["action_mask"]
    action = env.sample_valid_action()

    next_state, reward, terminated, truncated, next_info = env.step(action)

    replay_buffer.add(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminated=terminated,
        truncated=truncated,
        action_mask=action_mask,
        next_action_mask=next_info["action_mask"],
    )

    state = next_state
    info = next_info

    if terminated or truncated:
        break

if replay_buffer.can_sample(batch_size=4):
    batch = replay_buffer.sample(batch_size=4)
    print("states:", batch.states.shape)
    print("actions:", batch.actions.shape)
    print("rewards:", batch.rewards.shape)
    print("next_action_masks:", batch.next_action_masks.shape)

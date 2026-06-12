"""
example_training.py

End-to-end wiring example:
    MechanicalProteinEnv -> ReplayBuffer -> DDQNAgent

Copy this file into your project after adding:
    model/environment_module/environment.py
    model/replay_buffer_module/replay_buffer.py
    model/agent_module/ddqn_agent.py
"""

from pathlib import Path
import json

from model.agent_module.ddqn_agent import DDQNAgent, DDQNConfig
from model.environment_module.environment import MechanicalProteinEnv
from model.replay_buffer_module.replay_buffer import ReplayBuffer


OUTPUT_DIR = Path("outputs/ddqn_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = MechanicalProteinEnv(
    initial_pdb_path="model/reward_module/wild_type.pdb",
    max_steps=5,
    # Strongly recommended after debugging:
    # mutable_positions=[38, 39, 40, 41, 42, 67, 68, 69, 70],
)

state, info = env.reset(seed=7)

agent = DDQNAgent(
    state_shape=state.shape,
    action_dim=env.action_space.n,
    config=DDQNConfig(
        hidden_dims=(256, 256),
        gamma=0.99,
        learning_rate=1e-4,

        # At most 16 transitions are placed on the GPU at once.
        # Four backward passes accumulate into one optimizer update.
        micro_batch_size=16,
        gradient_accumulation_steps=4,

        replay_warmup_size=1_000,
        target_sync_interval=250,
        max_grad_norm=10.0,
        huber_beta=1.0,

        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,

        device="auto",
        use_amp=False,  # Set True on CUDA after the base pipeline is stable.
        seed=7,
    ),
)

replay_buffer = ReplayBuffer(
    capacity=100_000,
    state_shape=state.shape,
    action_dim=env.action_space.n,
    seed=7,
    store_action_masks=True,
)

episode_logs = []
total_environment_steps = 0

for episode in range(500):
    state, info = env.reset()
    episode_reward = 0.0
    optimization_logs = []

    while True:
        action_mask = info["action_mask"]

        action = agent.select_action(
            state,
            action_mask=action_mask,
        )

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

        optimization_result = agent.optimize_from_replay_buffer(replay_buffer)
        if optimization_result is not None:
            optimization_logs.append(optimization_result.to_dict())

        state = next_state
        info = next_info
        episode_reward += float(reward)
        total_environment_steps += 1

        if terminated or truncated:
            break

    candidate_path = OUTPUT_DIR / f"episode_{episode:04d}.pdb"
    env.save_current_pose(candidate_path)

    episode_logs.append(
        {
            "episode": episode,
            "reward": episode_reward,
            "epsilon": agent.epsilon,
            "environment_steps": total_environment_steps,
            "optimization_steps": agent.optimization_steps,
            "sequence": info.get("sequence"),
            "candidate_pdb": str(candidate_path),
            "last_optimization": (
                None if not optimization_logs else optimization_logs[-1]
            ),
        }
    )

    if (episode + 1) % 25 == 0:
        agent.save_checkpoint(OUTPUT_DIR / "agent_checkpoint.pt")
        replay_buffer.save(OUTPUT_DIR / "replay_buffer.npz")
        (OUTPUT_DIR / "training_log.json").write_text(
            json.dumps(episode_logs, indent=2),
            encoding="utf-8",
        )

print("Training example completed.")

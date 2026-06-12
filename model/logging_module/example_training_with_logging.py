"""
example_training_with_logging.py

End-to-end DDQN training with persistent logs and PNG visualization.

Required existing files:
    model/environment_module/environment.py
    model/reward_module/reward_calculators.py
    model/replay_buffer_module/replay_buffer.py
    model/agent_module/ddqn_agent.py
    model/logging_module/training_logger.py
"""

from pathlib import Path

from model.agent_module.ddqn_agent import DDQNAgent, DDQNConfig
from model.environment_module.environment import MechanicalProteinEnv
from model.logging_module.training_logger import (
    TrainingLogger,
    TrainingLoggerConfig,
)
from model.replay_buffer_module.replay_buffer import ReplayBuffer


OUTPUT_DIR = Path("outputs/ddqn_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = MechanicalProteinEnv(
    initial_pdb_path="model/reward_module/wild_type.pdb",
    max_steps=5,

    # After the complete pipeline is stable, restrict mutations to a
    # mechanically meaningful region.
    # mutable_positions=[38, 39, 40, 41, 42, 67, 68, 69, 70],
)

state, info = env.reset(seed=7)

agent_config = DDQNConfig(
    hidden_dims=(256, 256),
    gamma=0.99,
    learning_rate=1e-4,
    weight_decay=0.0,

    # Low-memory GPU configuration:
    # GPU sees at most 16 transitions at a time.
    # Four micro-batches form one effective batch of 64 transitions.
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
    use_amp=False,
    seed=7,
)

agent = DDQNAgent(
    state_shape=state.shape,
    action_dim=env.action_space.n,
    config=agent_config,
)

replay_buffer = ReplayBuffer(
    capacity=100_000,
    state_shape=state.shape,
    action_dim=env.action_space.n,
    seed=7,
    store_action_masks=True,
)

logger = TrainingLogger(
    TrainingLoggerConfig(
        output_dir=OUTPUT_DIR,
        rolling_window=20,
        plot_every_episodes=10,
        save_step_records=True,
        save_optimization_records=True,

        # Requires:
        #     python -m pip install tensorboard
        # Then run:
        #     tensorboard --logdir outputs/ddqn_base/tensorboard
        enable_tensorboard=False,

        resume=True,
        gradient_clip_threshold=agent_config.max_grad_norm,
    )
)

total_environment_steps = 0

try:
    for episode in range(500):
        state, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while True:
            action_mask = info["action_mask"]

            action = agent.select_action(
                state,
                action_mask=action_mask,
            )

            next_state, reward, terminated, truncated, next_info = env.step(
                action
            )

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

            optimization_result = agent.optimize_from_replay_buffer(
                replay_buffer
            )
            if optimization_result is not None:
                logger.log_optimization(
                    optimization_result,
                    global_step=total_environment_steps,
                )

            logger.log_step(
                episode=episode,
                episode_step=episode_steps,
                global_step=total_environment_steps,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=next_info,
            )

            state = next_state
            info = next_info
            episode_reward += float(reward)
            episode_steps += 1
            total_environment_steps += 1

            if terminated or truncated:
                break

        candidate_path = OUTPUT_DIR / "candidates" / f"episode_{episode:04d}.pdb"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        env.save_current_pose(candidate_path)

        logger.end_episode(
            episode=episode,
            total_reward=episode_reward,
            episode_steps=episode_steps,
            epsilon=agent.epsilon,
            optimization_steps=agent.optimization_steps,
            info=info,
            extra={
                "candidate_pdb": str(candidate_path),
            },
        )

        if (episode + 1) % 25 == 0:
            agent.save_checkpoint(OUTPUT_DIR / "checkpoints" / "agent.pt")
            replay_buffer.save(
                OUTPUT_DIR / "checkpoints" / "replay_buffer.npz"
            )

finally:
    # Always refresh plots and flush TensorBoard events, including after an
    # exception or keyboard interrupt.
    logger.generate_plots()
    logger.close()

print("Training completed.")

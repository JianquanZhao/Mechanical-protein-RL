from __future__ import annotations

import argparse
import threading

import numpy as np
import pytest
import torch

from model.agent_module.ddqn_agent import DDQNAgent, DDQNConfig
from model.replay_buffer_module.replay_buffer import ReplayBatch, ReplayBuffer
from training import (
    apply_full_batch_shortcut,
    configure_training_mode,
    data_parallel_batch_plan,
    enable_data_parallel,
    load_resume_checkpoint,
    run_replay_optimization_event,
    save_agent_checkpoint,
    save_resume_checkpoint,
    should_run_optimizer_event,
    update_schedule_summary,
)


def test_full_batch_shortcut_disables_gradient_accumulation() -> None:
    args = argparse.Namespace(
        batch_size=128,
        micro_batch_size=4,
        gradient_accumulation_steps=4,
    )

    apply_full_batch_shortcut(args)

    assert args.micro_batch_size == 128
    assert args.gradient_accumulation_steps == 1


def test_four_gpu_batch_plan_is_32_samples_per_gpu() -> None:
    config = DDQNConfig(
        micro_batch_size=128,
        gradient_accumulation_steps=1,
    )

    plan = data_parallel_batch_plan(config, (0, 1, 2, 3))

    assert plan == {
        "gpu_count": 4,
        "effective_batch_size": 128,
        "batch_per_backward": 128,
        "gradient_accumulation_steps": 1,
        "per_gpu_batch_size": 32,
        "uneven_batch_remainder": 0,
    }


def test_multi_gpu_batch_must_use_every_gpu() -> None:
    config = DDQNConfig(
        micro_batch_size=2,
        gradient_accumulation_steps=1,
    )

    with pytest.raises(ValueError, match="at least the number of GPUs"):
        data_parallel_batch_plan(config, (0, 1, 2, 3))


@pytest.mark.parametrize(
    ("completed_steps", "train_frequency", "expected"),
    [
        (1, 1, True),
        (1, 4, False),
        (3, 4, False),
        (4, 4, True),
        (8, 4, True),
        (16, 8, True),
    ],
)
def test_optimizer_event_frequency(
    completed_steps: int,
    train_frequency: int,
    expected: bool,
) -> None:
    assert should_run_optimizer_event(completed_steps, train_frequency) is expected


def test_update_schedule_summary_reports_independent_ratios() -> None:
    assert update_schedule_summary(
        batch_size=128,
        train_frequency=8,
        gradient_steps=2,
    ) == {
        "train_frequency": 8,
        "gradient_steps": 2,
        "gradient_updates_per_transition": 0.25,
        "replay_samples_per_transition": 32.0,
    }


def test_replay_optimization_event_runs_requested_gradient_steps() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.calls = 0

        def optimize_from_replay_buffer(self, _replay_buffer):
            self.calls += 1
            return f"optimization-{self.calls}"

    agent = FakeAgent()
    results = run_replay_optimization_event(
        agent=agent,
        replay_buffer=object(),
        gradient_steps=3,
    )

    assert agent.calls == 3
    assert results == ["optimization-1", "optimization-2", "optimization-3"]


def test_replay_optimization_event_stops_during_warmup() -> None:
    class WarmingAgent:
        def __init__(self) -> None:
            self.calls = 0

        def optimize_from_replay_buffer(self, _replay_buffer):
            self.calls += 1
            return None

    agent = WarmingAgent()
    results = run_replay_optimization_event(
        agent=agent,
        replay_buffer=object(),
        gradient_steps=3,
    )

    assert agent.calls == 1
    assert results == []


def test_replay_optimization_event_performs_real_ddqn_updates() -> None:
    replay_buffer = ReplayBuffer(
        capacity=8,
        state_shape=(2,),
        action_dim=2,
        seed=17,
        store_action_masks=True,
    )
    action_mask = np.ones(2, dtype=np.bool_)
    for index in range(4):
        state = np.asarray([index, index + 1], dtype=np.float32)
        replay_buffer.add(
            state=state,
            action=index % 2,
            reward=float(index) / 4.0,
            next_state=state + 0.25,
            terminated=index == 3,
            action_mask=action_mask,
            next_action_mask=action_mask,
        )

    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        online_network=torch.nn.Linear(2, 2),
        config=DDQNConfig(
            micro_batch_size=4,
            gradient_accumulation_steps=1,
            replay_warmup_size=4,
            device="cpu",
            seed=17,
        ),
    )

    results = run_replay_optimization_event(
        agent=agent,
        replay_buffer=replay_buffer,
        gradient_steps=3,
    )

    assert [result.optimization_step for result in results] == [1, 2, 3]
    assert agent.optimization_steps == 3


def test_resume_checkpoint_round_trip_restores_agent_replay_and_episode(tmp_path) -> None:
    config = DDQNConfig(
        hidden_dims=(8,),
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        replay_warmup_size=2,
        device="cpu",
        seed=17,
    )
    agent = DDQNAgent(state_shape=(2,), action_dim=2, config=config)
    agent.environment_steps = 12
    agent.optimization_steps = 3

    replay_buffer = ReplayBuffer(
        capacity=8,
        state_shape=(2,),
        action_dim=2,
        seed=17,
        store_action_masks=True,
    )
    action_mask = np.ones(2, dtype=np.bool_)
    for index in range(3):
        state = np.asarray([index, index + 1], dtype=np.float32)
        replay_buffer.add(
            state=state,
            action=index % 2,
            reward=float(index),
            next_state=state + 0.5,
            truncated=index == 2,
            action_mask=action_mask,
            next_action_mask=action_mask,
        )

    args = argparse.Namespace(
        dataset_seed=7,
        train_batch_size=8,
        no_shuffle_train=False,
        pdb_dir="/tmp/pdbs",
        train_index="/tmp/train_index.txt",
        resume_next_episode=None,
    )
    checkpoint_dir = tmp_path / "resume"
    save_resume_checkpoint(
        agent=agent,
        replay_buffer=replay_buffer,
        checkpoint_dir=checkpoint_dir,
        next_episode=5,
        args=args,
    )

    restored_agent, restored_replay, next_episode = load_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        agent_config=config,
        expected_state_shape=(2,),
        expected_action_dim=2,
        args=args,
    )

    assert next_episode == 5
    assert restored_agent.environment_steps == 12
    assert restored_agent.optimization_steps == 3
    assert len(restored_replay) == 3
    assert restored_replay.position == 3
    assert restored_replay.capacity == 8


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA device",
)
def test_float16_gradient_overflow_reduces_scale_and_retries() -> None:
    batch_size = 128
    states = np.ones((batch_size, 2), dtype=np.float32)
    dones = np.ones(batch_size, dtype=np.bool_)
    batch = ReplayBatch(
        states=states,
        actions=np.zeros(batch_size, dtype=np.int64),
        rewards=np.full(batch_size, 100.0, dtype=np.float32),
        next_states=states.copy(),
        terminateds=dones.copy(),
        truncateds=np.zeros_like(dones),
        dones=dones,
        action_masks=np.ones((batch_size, 2), dtype=np.bool_),
        next_action_masks=np.ones((batch_size, 2), dtype=np.bool_),
        indices=np.arange(batch_size, dtype=np.int64),
    )
    network = torch.nn.Linear(2, 2)
    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        online_network=network,
        config=DDQNConfig(
            micro_batch_size=batch_size,
            gradient_accumulation_steps=1,
            replay_warmup_size=0,
            device="cuda:0",
            use_amp=True,
            amp_dtype="float16",
            amp_max_retries=4,
            seed=17,
        ),
    )

    # This is the scale reached after 4,000 successful updates with PyTorch's
    # default initial scale and growth interval, matching the failed run.
    initial_scale = float(2**18)
    try:
        agent._grad_scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=initial_scale,
            growth_interval=2_000,
        )
    except (AttributeError, TypeError):
        agent._grad_scaler = torch.cuda.amp.GradScaler(
            init_scale=initial_scale,
            growth_interval=2_000,
        )

    result = agent.optimize_batch(batch)

    assert result.optimization_step == 1
    assert result.amp_retries >= 1
    assert result.amp_scale is not None
    assert result.amp_scale < initial_scale
    assert np.isfinite(result.grad_norm)

    payload = agent.state_dict()
    assert payload["grad_scaler"]["scale"] == result.amp_scale

    restored = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        online_network=torch.nn.Linear(2, 2),
        config=agent.config,
    )
    restored.load_state_dict(payload)
    assert restored._grad_scaler.get_scale() == result.amp_scale


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="requires four visible CUDA devices",
)
def test_four_gpu_full_batch_optimization_and_checkpoint(tmp_path) -> None:
    batch_size = 128
    residue_count = 16
    embedding_dim = 1280
    action_dim = residue_count * 20
    rng = np.random.default_rng(17)

    states = rng.standard_normal(
        (batch_size, residue_count, embedding_dim),
        dtype=np.float32,
    )
    next_states = rng.standard_normal(
        (batch_size, residue_count, embedding_dim),
        dtype=np.float32,
    )
    actions = rng.integers(0, action_dim, size=batch_size, dtype=np.int64)
    rewards = rng.uniform(-1.0, 1.0, size=batch_size).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.bool_)
    dones[::16] = True
    masks = np.ones((batch_size, action_dim), dtype=np.bool_)
    batch = ReplayBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        terminateds=dones.copy(),
        truncateds=np.zeros_like(dones),
        dones=dones,
        action_masks=masks.copy(),
        next_action_masks=masks.copy(),
        indices=np.arange(batch_size, dtype=np.int64),
    )

    device, gpu_ids = configure_training_mode(
        argparse.Namespace(
            mode="multi",
            device="auto",
            gpu_ids="0,1,2,3",
        )
    )
    assert device == "cuda:0"
    assert gpu_ids == (0, 1, 2, 3)

    agent = DDQNAgent(
        state_shape=(residue_count, embedding_dim),
        action_dim=action_dim,
        config=DDQNConfig(
            hidden_dims=(256, 256),
            embedding_dim=embedding_dim,
            micro_batch_size=batch_size,
            gradient_accumulation_steps=1,
            replay_warmup_size=0,
            target_sync_interval=1,
            device=device,
            use_amp=True,
            amp_dtype="bfloat16",
            seed=17,
        ),
    )
    enable_data_parallel(agent, gpu_ids)

    seen_devices: set[str] = set()
    seen_devices_lock = threading.Lock()

    def record_forward_device(_module, _inputs, output) -> None:
        with seen_devices_lock:
            seen_devices.add(str(output.device))

    hook = agent.online_network.module.register_forward_hook(record_forward_device)
    try:
        result = agent.optimize_batch(batch)
        for device_id in range(4):
            torch.cuda.synchronize(device_id)
    finally:
        hook.remove()

    assert isinstance(agent.online_network, torch.nn.DataParallel)
    assert isinstance(agent.target_network, torch.nn.DataParallel)
    assert result.effective_batch_size == 128
    assert result.micro_batches == 1
    assert result.optimization_step == 1
    assert result.target_synced is True
    assert result.amp_retries == 0
    assert result.amp_scale is None
    assert seen_devices == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}

    checkpoint_path = tmp_path / "agent.pt"
    save_agent_checkpoint(agent, checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert not any(key.startswith("module.") for key in payload["online_network"])
    assert not any(key.startswith("module.") for key in payload["target_network"])

    del agent
    torch.cuda.empty_cache()

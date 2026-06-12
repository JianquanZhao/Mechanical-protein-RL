from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model.agent_module.ddqn_agent import (
    DDQNAgent,
    DDQNConfig,
    QNetwork,
)


@dataclass
class SimpleBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_action_masks: np.ndarray | None


class LookupNetwork(nn.Module):
    """Return table rows selected by the first state feature."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))
        # Keep one parameter so DDQNAgent can construct an optimizer.
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, states):
        indices = states[:, 0].long()
        return self.table[indices] + self.dummy * 0.0


class TinyTrainableNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim)

    def forward(self, states):
        return self.linear(states)


class FakeReplayBuffer:
    def __init__(self, batch: SimpleBatch, size: int):
        self.batch = batch
        self.size = size
        self.sampled_batch_sizes = []

    def __len__(self):
        return self.size

    def sample(self, batch_size: int):
        self.sampled_batch_sizes.append(batch_size)
        assert batch_size == self.batch.actions.shape[0]
        return self.batch


def make_config(**kwargs) -> DDQNConfig:
    defaults = dict(
        hidden_dims=(8,),
        gamma=0.5,
        learning_rate=1e-3,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        replay_warmup_size=0,
        target_sync_interval=100,
        max_grad_norm=10.0,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=10,
        device="cpu",
        seed=7,
    )
    defaults.update(kwargs)
    return DDQNConfig(**defaults)


def make_batch(
    n: int = 4,
    *,
    state_dim: int = 2,
    action_dim: int = 2,
) -> SimpleBatch:
    rng = np.random.default_rng(17)
    states = rng.normal(size=(n, state_dim)).astype(np.float32)
    next_states = rng.normal(size=(n, state_dim)).astype(np.float32)
    actions = np.arange(n, dtype=np.int64) % action_dim
    rewards = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    dones = np.asarray([False] * (n - 1) + [True], dtype=bool)
    next_action_masks = np.ones((n, action_dim), dtype=bool)
    next_action_masks[-1] = False
    return SimpleBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        next_action_masks=next_action_masks,
    )


def clone_network(network: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(network)


def test_q_network_output_shape_and_single_state_support() -> None:
    network = QNetwork((6,), action_dim=5, hidden_dims=(8, 4))
    assert network(torch.zeros(3, 6)).shape == (3, 5)
    assert network(torch.zeros(6)).shape == (1, 5)


def test_q_network_rejects_wrong_state_shape() -> None:
    network = QNetwork((6,), action_dim=5, hidden_dims=(8,))
    with pytest.raises(ValueError, match="Expected trailing state shape"):
        network(torch.zeros(3, 5))


def test_epsilon_schedule_is_linear_and_bounded() -> None:
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=2,
        config=make_config(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100,
        ),
    )
    assert agent.epsilon == pytest.approx(1.0)
    agent.environment_steps = 50
    assert agent.epsilon == pytest.approx(0.55)
    agent.environment_steps = 1_000
    assert agent.epsilon == pytest.approx(0.1)


def test_select_action_greedy_respects_mask() -> None:
    online = LookupNetwork([[1.0, 100.0, 3.0]])
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=3,
        config=make_config(),
        online_network=online,
    )
    action = agent.select_action(
        np.asarray([0.0], dtype=np.float32),
        action_mask=np.asarray([True, False, True]),
        evaluate=True,
    )
    assert action == 2


def test_select_action_random_respects_mask() -> None:
    online = LookupNetwork([[1.0, 2.0, 3.0]])
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=3,
        config=make_config(
            epsilon_start=1.0,
            epsilon_end=1.0,
        ),
        online_network=online,
    )
    for _ in range(20):
        action = agent.select_action(
            np.asarray([0.0], dtype=np.float32),
            action_mask=np.asarray([False, True, False]),
            advance_step=False,
        )
        assert action == 1


def test_select_action_rejects_empty_mask() -> None:
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=2,
        config=make_config(),
    )
    with pytest.raises(ValueError, match="No valid action"):
        agent.select_action(
            np.asarray([0.0], dtype=np.float32),
            action_mask=np.asarray([False, False]),
        )


def test_double_dqn_target_uses_online_for_selection_target_for_evaluation() -> None:
    online = LookupNetwork(
        [
            [1.0, 5.0, 2.0],
            [9.0, 2.0, 1.0],
        ]
    )
    target = LookupNetwork(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=3,
        config=make_config(gamma=0.5),
        online_network=online,
        target_network=target,
    )

    next_states = torch.tensor([[0.0], [1.0]])
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([False, False])
    masks = torch.tensor(
        [
            [True, False, True],
            [True, True, True],
        ]
    )

    actual = agent.compute_ddqn_targets(
        next_states=next_states,
        rewards=rewards,
        dones=dones,
        next_action_masks=masks,
    )

    # Row 0: online chooses action 2 after masking; target evaluates 30.
    # Row 1: online chooses action 0; target evaluates 40.
    expected = torch.tensor([1.0 + 0.5 * 30.0, 2.0 + 0.5 * 40.0])
    torch.testing.assert_close(actual, expected)


def test_terminal_row_accepts_all_false_next_action_mask() -> None:
    online = LookupNetwork([[1.0, 2.0]])
    target = LookupNetwork([[10.0, 20.0]])
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=2,
        config=make_config(gamma=0.9),
        online_network=online,
        target_network=target,
    )

    actual = agent.compute_ddqn_targets(
        next_states=torch.tensor([[0.0]]),
        rewards=torch.tensor([7.5]),
        dones=torch.tensor([True]),
        next_action_masks=torch.tensor([[False, False]]),
    )

    torch.testing.assert_close(actual, torch.tensor([7.5]))


def test_non_terminal_row_rejects_all_false_next_action_mask() -> None:
    online = LookupNetwork([[1.0, 2.0]])
    agent = DDQNAgent(
        state_shape=(1,),
        action_dim=2,
        config=make_config(),
        online_network=online,
    )

    with pytest.raises(ValueError, match="non-terminal"):
        agent.compute_ddqn_targets(
            next_states=torch.tensor([[0.0]]),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([False]),
            next_action_masks=torch.tensor([[False, False]]),
        )


def test_gradient_accumulation_matches_single_micro_batch_update() -> None:
    torch.manual_seed(11)
    initial = TinyTrainableNetwork(state_dim=2, action_dim=2)
    batch = make_batch(n=4, state_dim=2, action_dim=2)

    accumulated = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        ),
        online_network=clone_network(initial),
    )
    full_batch = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(
            micro_batch_size=4,
            gradient_accumulation_steps=1,
        ),
        online_network=clone_network(initial),
    )

    accumulated_result = accumulated.optimize_batch(batch)
    full_batch_result = full_batch.optimize_batch(batch)

    assert accumulated_result.micro_batches == 2
    assert full_batch_result.micro_batches == 1
    assert accumulated_result.effective_batch_size == 4
    assert accumulated_result.loss == pytest.approx(full_batch_result.loss, rel=1e-6)

    for accumulated_parameter, full_parameter in zip(
        accumulated.online_network.parameters(),
        full_batch.online_network.parameters(),
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            full_parameter,
            rtol=1e-5,
            atol=1e-7,
        )


def test_optimize_from_replay_buffer_uses_effective_batch_size() -> None:
    batch = make_batch(n=6)
    buffer = FakeReplayBuffer(batch=batch, size=6)
    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(
            micro_batch_size=2,
            gradient_accumulation_steps=3,
            replay_warmup_size=0,
        ),
    )

    result = agent.optimize_from_replay_buffer(buffer)

    assert result is not None
    assert result.effective_batch_size == 6
    assert result.micro_batches == 3
    assert buffer.sampled_batch_sizes == [6]


def test_optimize_from_replay_buffer_waits_for_warmup() -> None:
    batch = make_batch(n=4)
    buffer = FakeReplayBuffer(batch=batch, size=4)
    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(
            replay_warmup_size=10,
        ),
    )

    assert agent.optimize_from_replay_buffer(buffer) is None
    assert buffer.sampled_batch_sizes == []


def test_target_network_hard_sync_interval() -> None:
    torch.manual_seed(5)
    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(target_sync_interval=1),
    )
    batch = make_batch(n=4)

    result = agent.optimize_batch(batch)

    assert result.target_synced
    for online_parameter, target_parameter in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
    ):
        torch.testing.assert_close(online_parameter, target_parameter)


def test_soft_target_update() -> None:
    torch.manual_seed(1)
    online = TinyTrainableNetwork(2, 2)
    target = TinyTrainableNetwork(2, 2)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(2.0)
        for parameter in target.parameters():
            parameter.fill_(0.0)

    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(),
        online_network=online,
        target_network=target,
    )
    agent.soft_sync_target_network(tau=0.25)

    for parameter in agent.target_network.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(
            epsilon_start=0.8,
            epsilon_end=0.2,
            epsilon_decay_steps=100,
        ),
    )
    batch = make_batch(n=4)
    _ = agent.optimize_batch(batch)
    agent.environment_steps = 17

    checkpoint_path = tmp_path / "agent.pt"
    agent.save_checkpoint(checkpoint_path)

    restored = DDQNAgent.from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    assert restored.environment_steps == 17
    assert restored.optimization_steps == agent.optimization_steps
    assert restored.epsilon == pytest.approx(agent.epsilon)

    states = torch.tensor([[0.2, -0.1], [0.7, 0.3]])
    with torch.no_grad():
        expected = agent.online_network(states)
        actual = restored.online_network(states)
    torch.testing.assert_close(actual, expected)


def test_gradient_clipping_returns_finite_norm() -> None:
    torch.manual_seed(2)
    batch = make_batch(n=4)
    batch.rewards[:] = 1_000_000.0

    agent = DDQNAgent(
        state_shape=(2,),
        action_dim=2,
        config=make_config(max_grad_norm=0.1),
    )
    result = agent.optimize_batch(batch)

    assert np.isfinite(result.grad_norm)
    # clip_grad_norm_ returns the norm before clipping, which is expected to
    # exceed the configured threshold for this intentionally extreme batch.
    assert result.grad_norm > 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"micro_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"target_sync_interval": 0},
        {"max_grad_norm": 0.0},
    ],
)
def test_config_validation(kwargs) -> None:
    config = make_config(**kwargs)
    with pytest.raises(ValueError):
        config.validate()

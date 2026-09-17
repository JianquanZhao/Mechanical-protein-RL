import numpy as np
import torch

from model.agent_module.ddqn_agent import QNetwork
from model.sft_module.dataset import SFTStateExample, collate_sft_states
from model.sft_module.losses import SFTLossConfig, compute_sft_loss
from model.sft_module.trainer import evaluate, train_epoch


def test_variable_length_collate_and_loss() -> None:
    examples = [
        SFTStateExample(
            state_id="s1",
            protein_id="p1",
            cluster_id="c1",
            split="train",
            sequence="AC",
            visited_positions=(),
            action_indices=np.asarray([1, 2]),
            reward_targets=np.asarray([1.0, -1.0], dtype=np.float32),
            reward_means=np.asarray([1.1, -0.8], dtype=np.float32),
            confidence_weights=np.asarray([1.0, 0.8], dtype=np.float32),
            label_classes=("confirmed_positive", "confirmed_negative"),
            strength_deltas=np.asarray([0.5, -0.5], dtype=np.float32),
            toughness_deltas=np.asarray([0.4, -0.4], dtype=np.float32),
        ),
        SFTStateExample(
            state_id="s2",
            protein_id="p2",
            cluster_id="c2",
            split="train",
            sequence="ACD",
            visited_positions=(2,),
            action_indices=np.asarray([3, 4]),
            reward_targets=np.asarray([0.5, -0.5], dtype=np.float32),
            reward_means=np.asarray([0.6, -0.4], dtype=np.float32),
            confidence_weights=np.asarray([1.0, 1.0], dtype=np.float32),
            label_classes=("confirmed_positive", "hard_negative"),
            strength_deltas=np.asarray([0.2, -0.2], dtype=np.float32),
            toughness_deltas=np.asarray([0.1, -0.1], dtype=np.float32),
        ),
    ]
    batch = collate_sft_states(examples)
    assert batch["targets"].shape == (2, 60)
    assert not batch["valid_mask"][1, 20:40].any()
    q_values = torch.zeros((2, 60), requires_grad=True)
    losses = compute_sft_loss(q_values, batch, SFTLossConfig())
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert q_values.grad is not None


def test_sft_train_and_evaluate_with_frozen_encoder_stub() -> None:
    example = SFTStateExample(
        state_id="s1",
        protein_id="p1",
        cluster_id="c1",
        split="train",
        sequence="AC",
        visited_positions=(),
        action_indices=np.asarray([1, 2]),
        reward_targets=np.asarray([1.0, -1.0], dtype=np.float32),
        reward_means=np.asarray([1.0, -1.0], dtype=np.float32),
        confidence_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        label_classes=("confirmed_positive", "confirmed_negative"),
        strength_deltas=np.asarray([0.5, -0.5], dtype=np.float32),
        toughness_deltas=np.asarray([0.4, -0.4], dtype=np.float32),
    )
    batch = collate_sft_states([example])

    class EncoderStub:
        @staticmethod
        def encode_sequences(sequences):
            return [np.zeros((len(sequence), 1280), dtype=np.float32) for sequence in sequences]

    model = QNetwork((1, 1281), 20, hidden_dims=(8,), embedding_dim=1280)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_metrics = train_epoch(
        model=model,
        encoder=EncoderStub(),
        loader=[batch],
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        loss_config=SFTLossConfig(),
        include_visited=True,
        gradient_clip=5.0,
    )
    validation_metrics = evaluate(
        model=model,
        encoder=EncoderStub(),
        loader=[batch],
        device=torch.device("cpu"),
        loss_config=SFTLossConfig(),
        include_visited=True,
    )
    assert np.isfinite(train_metrics["loss"])
    assert np.isfinite(validation_metrics["loss"])
    assert validation_metrics["state_count"] == 1.0

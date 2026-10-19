import numpy as np
import pandas as pd
import torch

from model.agent_module.ddqn_agent import QNetwork
from model.sft_module.compatibility import validate_sft_ddqn_q_network_compatibility
from model.sft_module.dataset import (
    SFTActionDataset,
    SFTStateExample,
    SFTStateReference,
    collate_sft_states,
)
from model.sft_module.losses import SFTLossConfig, compute_sft_loss
from model.sft_module.trainer import evaluate, train_epoch
from train_sft import main as train_sft_main


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


def test_sft_q_head_is_strictly_compatible_with_variable_length_ddqn() -> None:
    contract = validate_sft_ddqn_q_network_compatibility(
        embedding_dim=1280,
        hidden_dims=(256, 256),
        include_visited_mask=True,
        probe_residue_count=43,
    )
    assert contract["layer_dimensions"] == [1281, 256, 256, 20]
    assert contract["trainable_parameter_count"] == 399124
    assert contract["probe_output_shape"] == [2, 860]
    assert contract["strict_state_dict_load"] is True


def test_compact_dataset_keeps_only_references_until_collate(tmp_path) -> None:
    state_dir = tmp_path / "state_data"
    embedding_dir = tmp_path / "embeddings"
    state_dir.mkdir()
    embedding_dir.mkdir()
    np.savez_compressed(
        state_dir / "s1.npz",
        targets=np.asarray([[np.nan, 0.0, 0.5] + [np.nan] * 17], dtype=np.float32),
        reward_means=np.asarray([[np.nan, 0.005, 0.5] + [np.nan] * 17], dtype=np.float32),
        confidence_weights=np.asarray([[0.0, 0.25, 1.0] + [0.0] * 17], dtype=np.float32),
        label_codes=np.asarray([[0, 3, 1] + [0] * 17], dtype=np.int8),
        strength_deltas=np.asarray([[np.nan, 0.0, 0.2] + [np.nan] * 17], dtype=np.float32),
        toughness_deltas=np.asarray([[np.nan, 0.0, 0.1] + [np.nan] * 17], dtype=np.float32),
        observed_mask=np.asarray([[False, True, True] + [False] * 17]),
    )
    np.save(embedding_dir / "s1.npy", np.ones((1, 4), dtype=np.float16))
    pd.DataFrame(
        [
            {
                "state_id": "s1",
                "protein_id": "p1",
                "cluster_id": "c1",
                "split": "train",
                "sequence": "A",
                "visited_positions": "",
                "state_data_path": "state_data/s1.npz",
                "embedding_path": "embeddings/s1.npy",
                "embedding_dim": 4,
                "embedding_dtype": "float16",
            }
        ]
    ).to_csv(tmp_path / "states.csv", index=False)
    (tmp_path / "dataset_version.json").write_text(
        '{"format_version": "compact_state_v1"}\n', encoding="utf-8"
    )

    dataset = SFTActionDataset(
        tmp_path,
        split="train",
        expected_embedding_dim=4,
        require_cached_embeddings=True,
    )
    assert isinstance(dataset[0], SFTStateReference)
    batch = collate_sft_states([dataset[0]])
    assert batch["embeddings"].shape == (1, 1, 4)
    assert batch["targets"][0, 1].item() == 0.0
    assert not batch["positive_mask"][0, 1]
    assert batch["positive_mask"][0, 2]


def test_train_sft_runs_from_cached_compact_release(tmp_path) -> None:
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        state_id = f"state_{split}"
        state_path = tmp_path / "state_data" / f"{state_id}.npz"
        embedding_path = tmp_path / "embeddings" / f"{state_id}.npy"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "targets": np.full((2, 20), np.nan, dtype=np.float32),
            "reward_means": np.full((2, 20), np.nan, dtype=np.float32),
            "confidence_weights": np.zeros((2, 20), dtype=np.float32),
            "label_codes": np.zeros((2, 20), dtype=np.int8),
            "strength_deltas": np.full((2, 20), np.nan, dtype=np.float32),
            "toughness_deltas": np.full((2, 20), np.nan, dtype=np.float32),
            "observed_mask": np.zeros((2, 20), dtype=np.bool_),
        }
        for action_index, target, code in ((1, 0.5, 1), (2, -0.5, 2)):
            position, amino_acid = divmod(action_index, 20)
            arrays["targets"][position, amino_acid] = target
            arrays["reward_means"][position, amino_acid] = target
            arrays["confidence_weights"][position, amino_acid] = 1.0
            arrays["label_codes"][position, amino_acid] = code
            arrays["strength_deltas"][position, amino_acid] = target
            arrays["toughness_deltas"][position, amino_acid] = target
            arrays["observed_mask"][position, amino_acid] = True
        np.savez_compressed(state_path, **arrays)
        np.save(
            embedding_path,
            np.full((2, 1280), split_index, dtype=np.float16),
        )
        rows.append(
            {
                "state_id": state_id,
                "protein_id": f"protein_{split}",
                "cluster_id": f"cluster_{split}",
                "split": split,
                "sequence": "AC",
                "visited_positions": "",
                "state_data_path": str(state_path.relative_to(tmp_path)),
                "embedding_path": str(embedding_path.relative_to(tmp_path)),
                "embedding_dim": 1280,
                "embedding_dtype": "float16",
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "states.csv", index=False)
    (tmp_path / "dataset_version.json").write_text(
        '{"format_version":"compact_state_v1",'
        '"embedding_cache":{"complete":true,"embedding_dim":1280}}\n',
        encoding="utf-8",
    )

    result = train_sft_main(
        [
            "--data-dir",
            str(tmp_path),
            "--run-id",
            "compact_smoke",
            "--model-root",
            str(tmp_path / "model"),
            "--log-root",
            str(tmp_path / "log"),
            "--tensorboard-root",
            str(tmp_path / "tensorboard"),
            "--hidden-dims",
            "8",
            "--epochs",
            "1",
            "--state-batch-size",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert result == 0
    assert (tmp_path / "model" / "compact_smoke" / "best.pt").is_file()
    assert (tmp_path / "model" / "compact_smoke" / "run_summary.json").is_file()

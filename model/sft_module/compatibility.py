"""Compatibility checks between offline SFT and online DDQN Q heads."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from model.agent_module.ddqn_agent import AMINO_ACID_ACTION_DIM, QNetwork


def _parameter_shapes(model: QNetwork) -> Dict[str, list[int]]:
    return {
        name: list(parameter.shape)
        for name, parameter in model.state_dict().items()
    }


def validate_sft_ddqn_q_network_compatibility(
    *,
    embedding_dim: int,
    hidden_dims: Sequence[int],
    include_visited_mask: bool,
    probe_residue_count: int = 37,
) -> Dict[str, Any]:
    """Prove that an SFT Q head loads strictly into a variable-length DDQN head."""

    embedding_dim = int(embedding_dim)
    hidden_dims = tuple(int(value) for value in hidden_dims)
    probe_residue_count = int(probe_residue_count)
    if probe_residue_count <= 1:
        raise ValueError("probe_residue_count must be greater than one.")

    feature_dim = embedding_dim + int(bool(include_visited_mask))
    sft_network = QNetwork(
        (1, feature_dim),
        AMINO_ACID_ACTION_DIM,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
    )
    ddqn_network = QNetwork(
        (probe_residue_count, feature_dim),
        probe_residue_count * AMINO_ACID_ACTION_DIM,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
    )

    if not sft_network.per_residue_mode or not ddqn_network.per_residue_mode:
        raise RuntimeError("SFT and DDQN must both use per-residue QNetwork mode.")
    sft_shapes = _parameter_shapes(sft_network)
    ddqn_shapes = _parameter_shapes(ddqn_network)
    if sft_shapes != ddqn_shapes:
        raise RuntimeError(
            "SFT and DDQN Q-head parameter shapes differ: "
            f"sft={sft_shapes}, ddqn={ddqn_shapes}."
        )

    # Strict loading is the strongest practical checkpoint compatibility test.
    ddqn_network.load_state_dict(sft_network.state_dict(), strict=True)
    with torch.inference_mode():
        output = ddqn_network(
            torch.zeros((2, probe_residue_count, feature_dim), dtype=torch.float32)
        )
    expected_output_shape = (2, probe_residue_count * AMINO_ACID_ACTION_DIM)
    if tuple(output.shape) != expected_output_shape:
        raise RuntimeError(
            f"DDQN Q output shape {tuple(output.shape)} != {expected_output_shape}."
        )

    return {
        "network_class": f"{QNetwork.__module__}.{QNetwork.__name__}",
        "mode": "per_residue_shared_head",
        "embedding_dim": embedding_dim,
        "include_visited_mask": bool(include_visited_mask),
        "state_feature_dim": feature_dim,
        "hidden_dims": list(hidden_dims),
        "per_residue_action_dim": AMINO_ACID_ACTION_DIM,
        "layer_dimensions": [feature_dim, *hidden_dims, AMINO_ACID_ACTION_DIM],
        "parameter_shapes": sft_shapes,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in sft_network.parameters()
        ),
        "probe_residue_count": probe_residue_count,
        "probe_output_shape": list(expected_output_shape),
        "strict_state_dict_load": True,
    }

"""Training and evaluation loops for a frozen-ESM2 per-residue Q head."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .losses import SFTLossConfig, compute_sft_loss
from .metrics import compute_sft_metrics


def encode_batch(encoder: Any, batch: Dict[str, Any], *, device: torch.device, include_visited: bool) -> Tensor:
    arrays = encoder.encode_sequences(batch["sequences"])
    max_length = int(batch["lengths"].max().item())
    embedding_dim = int(arrays[0].shape[1])
    states = torch.zeros((len(arrays), max_length, embedding_dim), dtype=torch.float32, device=device)
    for index, array in enumerate(arrays):
        states[index, : len(array)] = torch.from_numpy(array).to(device=device)
    if include_visited:
        visited = batch["visited"].to(device=device)
        states = torch.cat((states, visited), dim=-1)
    return states


def move_targets(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device=device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def train_epoch(
    *,
    model: nn.Module,
    encoder: Any,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    loss_config: SFTLossConfig,
    include_visited: bool,
    gradient_clip: float,
) -> Dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    steps = 0
    for raw_batch in loader:
        batch = move_targets(raw_batch, device)
        states = encode_batch(encoder, raw_batch, device=device, include_visited=include_visited)
        q_values = model(states)
        losses = compute_sft_loss(q_values, batch, loss_config)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        totals["grad_norm"] += float(grad_norm.detach().cpu())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    encoder: Any,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    loss_config: SFTLossConfig,
    include_visited: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    q_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    strength_rows: list[np.ndarray] = []
    toughness_rows: list[np.ndarray] = []
    max_actions = 0
    batches: list[tuple[np.ndarray, ...]] = []
    for raw_batch in loader:
        batch = move_targets(raw_batch, device)
        states = encode_batch(encoder, raw_batch, device=device, include_visited=include_visited)
        q_values = model(states)
        losses = compute_sft_loss(q_values, batch, loss_config)
        total_loss += float(losses["loss"].cpu())
        steps += 1
        arrays = (
            q_values.cpu().numpy(),
            batch["targets"].cpu().numpy(),
            batch["observed_mask"].cpu().numpy(),
            batch["positive_mask"].cpu().numpy(),
            batch["strength_deltas"].cpu().numpy(),
            batch["toughness_deltas"].cpu().numpy(),
        )
        max_actions = max(max_actions, arrays[0].shape[1])
        batches.append(arrays)

    def pad(array: np.ndarray, value: float | bool) -> np.ndarray:
        width = max_actions - array.shape[1]
        return np.pad(array, ((0, 0), (0, width)), constant_values=value)

    for arrays in batches:
        q_rows.append(pad(arrays[0], 0.0))
        target_rows.append(pad(arrays[1], np.nan))
        observed_rows.append(pad(arrays[2], False))
        positive_rows.append(pad(arrays[3], False))
        strength_rows.append(pad(arrays[4], np.nan))
        toughness_rows.append(pad(arrays[5], np.nan))
    metrics = compute_sft_metrics(
        np.concatenate(q_rows),
        np.concatenate(target_rows),
        np.concatenate(observed_rows),
        np.concatenate(positive_rows),
        np.concatenate(strength_rows),
        np.concatenate(toughness_rows),
    )
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


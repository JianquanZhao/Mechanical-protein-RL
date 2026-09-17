"""Listwise, pairwise-ranking and robust regression objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SFTLossConfig:
    listwise_weight: float = 1.0
    ranking_weight: float = 0.5
    regression_weight: float = 0.5
    target_temperature: float = 1.0
    q_temperature: float = 1.0
    ranking_margin: float = 0.25
    huber_beta: float = 1.0
    target_clip_min: float = -20.0
    target_clip_max: float = 20.0


def compute_sft_loss(q_values: Tensor, batch: Dict[str, Tensor], config: SFTLossConfig) -> Dict[str, Tensor]:
    observed = batch["observed_mask"]
    targets = batch["targets"].clamp(config.target_clip_min, config.target_clip_max)
    confidence = batch["confidence_weights"].clamp_min(0.0)
    if not torch.any(observed):
        raise ValueError("SFT batch contains no observed action labels.")

    regression_values = F.smooth_l1_loss(
        q_values[observed], targets[observed], reduction="none", beta=config.huber_beta
    )
    regression_weights = confidence[observed]
    regression = (regression_values * regression_weights).sum() / regression_weights.sum().clamp_min(1e-8)

    listwise_terms: list[Tensor] = []
    ranking_terms: list[Tensor] = []
    for index in range(q_values.shape[0]):
        mask = observed[index]
        if int(mask.sum()) >= 2:
            target_probabilities = torch.softmax(targets[index, mask] / config.target_temperature, dim=0)
            q_log_probabilities = torch.log_softmax(q_values[index, mask] / config.q_temperature, dim=0)
            listwise_terms.append(
                torch.sum(target_probabilities * (torch.log(target_probabilities.clamp_min(1e-12)) - q_log_probabilities))
            )
        positives = batch["positive_mask"][index] & mask
        negatives = batch["negative_mask"][index] & mask
        if torch.any(positives) and torch.any(negatives):
            positive_score = q_values[index, positives].max()
            negative_score = q_values[index, negatives].max()
            ranking_terms.append(F.relu(config.ranking_margin - positive_score + negative_score))

    zero = q_values.sum() * 0.0
    listwise = torch.stack(listwise_terms).mean() if listwise_terms else zero
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else zero
    total = (
        config.listwise_weight * listwise
        + config.ranking_weight * ranking
        + config.regression_weight * regression
    )
    return {"loss": total, "listwise_loss": listwise, "ranking_loss": ranking, "regression_loss": regression}


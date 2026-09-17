"""Train the DDQN per-residue Q head on versioned offline mutation labels."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.agent_module.ddqn_agent import QNetwork
from model.encoding_module.esm2_encoder import ESM2SequenceEncoder
from model.sft_module import SFTActionDataset, SFTLossConfig, collate_sft_states
from model.sft_module.trainer import evaluate, train_epoch


DEFAULT_ROOT = Path("/mnt/nas/jianquanzhao/data/mprl/SFT")
LOGGER = logging.getLogger("train_sft")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-version", default="sft_qhead_v001")
    parser.add_argument("--model-root", default=str(DEFAULT_ROOT / "model"))
    parser.add_argument("--log-root", default=str(DEFAULT_ROOT / "log"))
    parser.add_argument("--tensorboard-root", default=str(DEFAULT_ROOT / "tensorboard"))
    parser.add_argument("--esm-model-dir", default=None)
    parser.add_argument("--embedding-dim", type=int, choices=(1280, 2560, 5120), default=1280)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--include-visited-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-column", default="reward_lcb")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--state-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--q-temperature", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.25)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--target-clip-min", type=float, default=-20.0)
    parser.add_argument("--target-clip-max", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--enable-tensorboard", action="store_true")
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(path)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _checkpoint(
    *,
    path: Path,
    model: QNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    loss_config: SFTLossConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "mechanical_protein_sft_q_head",
            "q_network_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "architecture": {
                "embedding_dim": args.embedding_dim,
                "hidden_dims": [int(value) for value in args.hidden_dims.split(",")],
                "include_visited_mask": args.include_visited_mask,
                "amino_acid_order": "ACDEFGHIKLMNPQRSTVWY",
                "per_residue_action_dim": 20,
            },
            "target_column": args.target_column,
            "loss_config": asdict(loss_config),
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        },
        path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = f"{args.run_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = Path(args.model_root).expanduser().resolve() / run_id
    log_dir = Path(args.log_root).expanduser().resolve() / run_id
    tensorboard_dir = Path(args.tensorboard_root).expanduser().resolve() / run_id
    _configure_logging(log_dir / "train_sft.log")
    _seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_data = SFTActionDataset(args.data_dir, split="train", target_column=args.target_column)
    validation_data = SFTActionDataset(args.data_dir, split="validation", target_column=args.target_column)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.state_batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_sft_states,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)

    encoder = ESM2SequenceEncoder(
        embedding_dim=args.embedding_dim,
        device=str(device),
        mutable_only=True,
        model_dir=args.esm_model_dir,
    )
    feature_dim = args.embedding_dim + int(args.include_visited_mask)
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value.strip())
    model = QNetwork((1, feature_dim), 20, hidden_dims=hidden_dims, embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loss_config = SFTLossConfig(
        listwise_weight=args.listwise_weight,
        ranking_weight=args.ranking_weight,
        regression_weight=args.regression_weight,
        target_temperature=args.target_temperature,
        q_temperature=args.q_temperature,
        ranking_margin=args.ranking_margin,
        huber_beta=args.huber_beta,
        target_clip_min=args.target_clip_min,
        target_clip_max=args.target_clip_max,
    )
    start_epoch = 1
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["q_network_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1

    writer = None
    if args.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tensorboard_dir))
    LOGGER.info(
        "SFT run=%s device=%s train_states=%s validation_states=%s model_dir=%s",
        run_id,
        device,
        len(train_data),
        len(validation_data),
        model_dir,
    )
    best_ndcg = -float("inf")
    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            encoder=encoder,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_config=loss_config,
            include_visited=args.include_visited_mask,
            gradient_clip=args.gradient_clip,
        )
        validation_metrics = evaluate(
            model=model,
            encoder=encoder,
            loader=validation_loader,
            device=device,
            loss_config=loss_config,
            include_visited=args.include_visited_mask,
        )
        LOGGER.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_ndcg=%.6f val_spearman=%.6f "
            "val_hit1=%.6f val_greedy_reward=%.6f",
            epoch,
            train_metrics["loss"],
            validation_metrics["loss"],
            validation_metrics["ndcg_macro"],
            validation_metrics["spearman_macro"],
            validation_metrics["positive_hit_at_1"],
            validation_metrics["greedy_reward_mean"],
        )
        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in validation_metrics.items():
                writer.add_scalar(f"validation/{key}", value, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        _checkpoint(
            path=model_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            metrics=validation_metrics,
            loss_config=loss_config,
        )
        ndcg = float(validation_metrics["ndcg_macro"])
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            stale_epochs = 0
            _checkpoint(
                path=model_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                metrics=validation_metrics,
                loss_config=loss_config,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stopping_patience:
            LOGGER.info("Early stopping at epoch=%s best_validation_ndcg=%.6f", epoch, best_ndcg)
            break
    if writer is not None:
        writer.close()
    (model_dir / "run_summary.json").write_text(
        json.dumps({"run_id": run_id, "best_validation_ndcg": best_ndcg}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

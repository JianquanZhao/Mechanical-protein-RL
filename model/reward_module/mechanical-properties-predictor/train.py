from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset_module import (  # noqa: E402
    MechanicalPropertyDataset,
    TargetNormalizer,
    build_split,
    encode_records,
    load_mechanical_property_records,
)
from logging_module import PredictorLogger  # noqa: E402
from metric_module import compute_metrics, format_metrics  # noqa: E402
from model_module import MechanicalPropertyMLP, MechanicalPropertyMLPConfig  # noqa: E402


DEFAULT_CSV_PATH = "/mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv"
LOGGER = logging.getLogger("mechanical_property_predictor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ESM2 + MLP mechanical-property predictor.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", default="outputs/mechanical_property_predictor")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--toughness-column", default="v127")
    parser.add_argument("--strength-column", default="v128")
    parser.add_argument("--id-column", default="PDB_ID")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--split-method", choices=("random", "similarity"), default="random")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--embedding-dim", type=int, choices=(1280, 2560, 5120), default=1280)
    parser.add_argument("--esm2-device", default="auto")
    parser.add_argument("--esm2-batch-size", type=int, default=4)
    parser.add_argument("--embedding-cache-dir", default=None)

    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims must contain positive comma-separated integers.")
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def make_dataset(
    embeddings: np.ndarray,
    targets: np.ndarray,
    records,
    indices: list[int],
) -> MechanicalPropertyDataset:
    return MechanicalPropertyDataset(
        embeddings=embeddings[indices],
        targets=targets[indices],
        records=[records[index] for index in indices],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(embeddings)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    normalizer: TargetNormalizer,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions = []
    targets = []
    for embeddings, batch_targets in loader:
        embeddings = embeddings.to(device)
        batch_targets = batch_targets.to(device)
        batch_predictions = model(embeddings)
        loss = criterion(batch_predictions, batch_targets)
        batch_size = int(batch_targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        predictions.append(batch_predictions.detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())

    pred_norm = np.concatenate(predictions, axis=0)
    true_norm = np.concatenate(targets, axis=0)
    pred_raw = normalizer.inverse_transform(pred_norm)
    true_raw = normalizer.inverse_transform(true_norm)
    return total_loss / max(1, total_count), true_raw, pred_raw


def records_for_indices(records, indices: list[int]) -> list:
    return [records[index] for index in indices]


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)
    device = resolve_device(args.device)
    LOGGER.info("Starting predictor training args=%s device=%s", vars(args), device)

    records = load_mechanical_property_records(
        args.csv_path,
        sequence_column=args.sequence_column,
        toughness_column=args.toughness_column,
        strength_column=args.strength_column,
        id_column=args.id_column,
        max_samples=args.max_samples,
    )
    split = build_split(
        records,
        split_method=args.split_method,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        kmer_size=args.kmer_size,
    )
    LOGGER.info("Split sizes: %s", {name: len(indices) for name, indices in split.items()})

    cache_dir = (
        Path(args.embedding_cache_dir)
        if args.embedding_cache_dir is not None
        else Path(args.output_dir) / "embedding_cache"
    )
    embeddings = encode_records(
        records,
        embedding_dim=args.embedding_dim,
        device=args.esm2_device,
        batch_size=args.esm2_batch_size,
        cache_dir=cache_dir,
    )

    raw_targets = np.stack([record.targets for record in records]).astype(np.float32)
    normalizer = TargetNormalizer.fit(raw_targets[split["train"]])
    normalized_targets = normalizer.transform(raw_targets)

    train_dataset = make_dataset(embeddings, normalized_targets, records, split["train"])
    val_dataset = make_dataset(embeddings, normalized_targets, records, split["val"])
    test_dataset = make_dataset(embeddings, normalized_targets, records, split["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = MechanicalPropertyMLPConfig(
        input_dim=args.embedding_dim,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
    )
    model = MechanicalPropertyMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    logger = PredictorLogger(args.output_dir, enable_tensorboard=args.enable_tensorboard)
    logger.write_split_indices(split)

    run_config = {
        "args": vars(args),
        "model_config": config.to_dict(),
        "target_normalizer": normalizer.to_dict(),
        "split_sizes": {name: len(indices) for name, indices in split.items()},
    }
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_path = logger.checkpoint_dir / "best_model.pt"
    last_path = logger.checkpoint_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, y_val_true, y_val_pred = evaluate(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            normalizer=normalizer,
        )
        val_metrics = compute_metrics(y_val_true, y_val_pred)
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics={f"val/{key}": value for key, value in val_metrics.items()},
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        LOGGER.info(
            "Epoch %s/%s train_loss=%.6f val_loss=%.6f %s",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            format_metrics({f"val/{key}": value for key, value in val_metrics.items()}),
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "target_normalizer": normalizer.to_dict(),
            "args": vars(args),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                LOGGER.info("Early stopping triggered at epoch=%s best_epoch=%s", epoch, best_epoch)
                break

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    split_metrics = {}
    prediction_payload = {}
    for split_name, dataset, loader in (
        ("train", train_dataset, DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)),
        ("val", val_dataset, val_loader),
        ("test", test_dataset, test_loader),
    ):
        loss, y_true, y_pred = evaluate(
            model,
            loader,
            criterion=criterion,
            device=device,
            normalizer=normalizer,
        )
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = float(loss)
        split_metrics[split_name] = metrics
        selected_records = records_for_indices(records, split[split_name])
        logger.write_predictions(
            split_name=split_name,
            record_ids=[record.record_id for record in selected_records],
            y_true=y_true,
            y_pred=y_pred,
        )
        prediction_payload[split_name] = {
            "loss": float(loss),
            "metrics": metrics,
        }
        LOGGER.info("%s metrics: %s", split_name, format_metrics(metrics))

    if args.split_method == "similarity":
        split_metrics["ood"] = split_metrics["test"]

    final_payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "split_metrics": split_metrics,
        "target_normalizer": normalizer.to_dict(),
    }
    logger.write_final_metrics(final_payload)
    logger.plot_history()
    logger.close()
    LOGGER.info("Training complete best_epoch=%s best_val_loss=%.6f output_dir=%s", best_epoch, best_val_loss, args.output_dir)


if __name__ == "__main__":
    main()

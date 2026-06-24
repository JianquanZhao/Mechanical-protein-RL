from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Mapping


LOGGER = logging.getLogger(__name__)


class PredictorLogger:
    def __init__(self, output_dir: str | Path, *, enable_tensorboard: bool = False) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.logs_dir = self.output_dir / "logs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.logs_dir / "history.jsonl"
        self.metrics_path = self.logs_dir / "metrics.json"
        self._writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(self.output_dir / "tensorboard"))
            except ImportError:
                LOGGER.warning("TensorBoard is unavailable; continuing without it.")

    def log_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Mapping[str, float],
        learning_rate: float,
    ) -> None:
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(learning_rate),
            **{key: float(value) for key, value in metrics.items()},
        }
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")
        if self._writer is not None:
            self._writer.add_scalar("loss/train", train_loss, epoch)
            self._writer.add_scalar("loss/val", val_loss, epoch)
            for key, value in metrics.items():
                self._writer.add_scalar(f"metrics/{key}", value, epoch)
            self._writer.flush()

    def write_final_metrics(self, payload: Mapping) -> None:
        self.metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_split_indices(self, split: Mapping[str, list[int]]) -> None:
        for name, indices in split.items():
            path = self.logs_dir / f"{name}_indices.txt"
            path.write_text("\n".join(str(index) for index in indices) + "\n", encoding="utf-8")

    def write_predictions(
        self,
        *,
        split_name: str,
        record_ids: list[str],
        y_true,
        y_pred,
    ) -> None:
        path = self.logs_dir / f"{split_name}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "record_id",
                    "true_strength",
                    "true_toughness",
                    "pred_strength",
                    "pred_toughness",
                ]
            )
            for record_id, true_row, pred_row in zip(record_ids, y_true, y_pred):
                writer.writerow([record_id, true_row[0], true_row[1], pred_row[0], pred_row[1]])

    def plot_history(self) -> None:
        if not self.history_path.is_file():
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("matplotlib is unavailable; skipping plots.")
            return
        records = [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line]
        if not records:
            return
        epochs = [record["epoch"] for record in records]
        self._plot_lines(
            epochs,
            {
                "train_loss": [record["train_loss"] for record in records],
                "val_loss": [record["val_loss"] for record in records],
            },
            self.plots_dir / "loss.png",
            ylabel="MSE loss",
        )
        for metric in ("mean/r2", "mean/mae", "mean/rmse", "mean/spearman"):
            if metric in records[-1]:
                self._plot_lines(
                    epochs,
                    {metric: [record[metric] for record in records]},
                    self.plots_dir / f"{metric.replace('/', '_')}.png",
                    ylabel=metric,
                )

    @staticmethod
    def _plot_lines(epochs, lines: Mapping[str, list[float]], path: Path, *, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, values in lines.items():
            ax.plot(epochs, values, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

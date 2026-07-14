"""Train and load the random-split hbond random forest reward artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


SELECTED_HBOND_FEATURES = (
    "sequence_length",
    "hbond_per_residue",
    "seq_class_nonlocal_per_residue",
    "strong_nonlocal_fraction",
    "strong_nonlocal_per_residue",
    "nonlocal_backbone_backbone_per_residue",
    "hbond_contact_order",
)
TARGET_COLUMNS = ("v127", "v128")

REWARD_MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RANDOM_SPLIT_DIR = (
    REWARD_MODULE_DIR
    / "mechanical-properties-predictor"
    / "outputs"
    / "hbond_analysis"
)
DEFAULT_HBOND_TABLE_PATH = DEFAULT_RANDOM_SPLIT_DIR / "hbond_length_disentanglement_table.csv"
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "hbond_random_forest.joblib"


@dataclass(frozen=True)
class HbondRandomForestArtifact:
    model: Any
    selected_features: Sequence[str]
    target_columns: Sequence[str]
    target_mean: Mapping[str, float]
    target_std: Mapping[str, float]
    metrics: Mapping[str, Mapping[str, float]]
    train_size: int
    val_size: int
    test_size: int
    source_table_path: str
    split_method: str = "random"
    seed: int = 7
    target_transform: str = "log1p"


def load_or_train_random_forest_artifact(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    table_path: str | Path = DEFAULT_HBOND_TABLE_PATH,
    split_dir: str | Path = DEFAULT_RANDOM_SPLIT_DIR,
    force_retrain: bool = False,
) -> HbondRandomForestArtifact:
    artifact_path = Path(artifact_path)
    if artifact_path.exists() and not force_retrain:
        return joblib.load(artifact_path)

    artifact = train_random_split_random_forest(table_path=table_path, split_dir=split_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def train_random_split_random_forest(
    *,
    table_path: str | Path = DEFAULT_HBOND_TABLE_PATH,
    split_dir: str | Path = DEFAULT_RANDOM_SPLIT_DIR,
) -> HbondRandomForestArtifact:
    table_path = Path(table_path)
    split_dir = Path(split_dir)
    data = pd.read_csv(table_path)
    data = data.dropna(subset=list(SELECTED_HBOND_FEATURES) + list(TARGET_COLUMNS) + ["PDB_ID"])

    train_ids = _read_split_ids(split_dir / "dataset_random_train_pdb_ids.txt")
    val_ids = _read_split_ids(split_dir / "dataset_random_val_pdb_ids.txt")
    test_ids = _read_split_ids(split_dir / "dataset_random_test_pdb_ids.txt")

    if not train_ids:
        train_ids, val_ids, test_ids = _fallback_random_ids(data["PDB_ID"].astype(str).to_numpy())

    train = data[data["PDB_ID"].astype(str).isin(train_ids)].copy()
    val = data[data["PDB_ID"].astype(str).isin(val_ids)].copy()
    test = data[data["PDB_ID"].astype(str).isin(test_ids)].copy()
    if train.empty:
        raise ValueError(f"No random-split training rows found in {table_path}.")

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=7,
            n_jobs=-1,
        )
    )

    x_train = train.loc[:, SELECTED_HBOND_FEATURES].to_numpy(dtype=float)
    y_train = np.log1p(train.loc[:, TARGET_COLUMNS].to_numpy(dtype=float))
    model.fit(x_train, y_train)

    target_mean = {
        "toughness": float(train["v127"].mean()),
        "strength": float(train["v128"].mean()),
    }
    target_std = {
        "toughness": float(train["v127"].std(ddof=0) or 1.0),
        "strength": float(train["v128"].std(ddof=0) or 1.0),
    }
    metrics = {
        "train": _score_split(model, train),
        "val": _score_split(model, val),
        "test": _score_split(model, test),
    }

    return HbondRandomForestArtifact(
        model=model,
        selected_features=tuple(SELECTED_HBOND_FEATURES),
        target_columns=tuple(TARGET_COLUMNS),
        target_mean=target_mean,
        target_std=target_std,
        metrics=metrics,
        train_size=int(len(train)),
        val_size=int(len(val)),
        test_size=int(len(test)),
        source_table_path=str(table_path),
    )


def predict_physical_units(artifact: HbondRandomForestArtifact, x: np.ndarray) -> np.ndarray:
    log_prediction = np.asarray(artifact.model.predict(x), dtype=float)
    return np.expm1(log_prediction)


def _score_split(model: Any, frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {"n": 0.0}
    x = frame.loc[:, SELECTED_HBOND_FEATURES].to_numpy(dtype=float)
    y_true = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    y_pred = np.expm1(np.asarray(model.predict(x), dtype=float))
    result: Dict[str, float] = {"n": float(len(frame))}
    names = ("toughness", "strength")
    for index, name in enumerate(names):
        result[f"{name}_r2"] = float(r2_score(y_true[:, index], y_pred[:, index]))
        result[f"{name}_mae"] = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        mse = mean_squared_error(y_true[:, index], y_pred[:, index])
        result[f"{name}_rmse"] = float(np.sqrt(mse))
    return result


def _read_split_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _fallback_random_ids(pdb_ids: Sequence[str]) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(7)
    ids = np.asarray(sorted(set(str(pdb_id) for pdb_id in pdb_ids)))
    rng.shuffle(ids)
    n_total = len(ids)
    n_test = int(round(n_total * 0.1))
    n_val = int(round(n_total * 0.1))
    test = set(ids[:n_test])
    val = set(ids[n_test : n_test + n_val])
    train = set(ids[n_test + n_val :])
    return train, val, test

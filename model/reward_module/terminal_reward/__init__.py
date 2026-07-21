"""Terminal reward calculators for structure-based mechanical-property reward."""

from .calculator import (
    DualStructureTerminalRewardResult,
    EqualWeightDualStructureTerminalRewardCalculator,
    HbondTopologyTerminalRewardCalculator,
    StructureRewardResult,
    TerminalRewardScalarization,
)
from .features import HbondFeatureExtractor, HbondFeatureResult
from .model_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_HBOND_TABLE_PATH,
    DEFAULT_RANDOM_SPLIT_DIR,
    SELECTED_HBOND_FEATURES,
    load_or_train_random_forest_artifact,
)

__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_HBOND_TABLE_PATH",
    "DEFAULT_RANDOM_SPLIT_DIR",
    "DualStructureTerminalRewardResult",
    "EqualWeightDualStructureTerminalRewardCalculator",
    "HbondFeatureExtractor",
    "HbondFeatureResult",
    "HbondTopologyTerminalRewardCalculator",
    "SELECTED_HBOND_FEATURES",
    "StructureRewardResult",
    "TerminalRewardScalarization",
    "load_or_train_random_forest_artifact",
]

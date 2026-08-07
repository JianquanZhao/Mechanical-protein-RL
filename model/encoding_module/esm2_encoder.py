"""
ESM2 sequence encoder for MechanicalProteinEnv observations.

The encoder is intentionally lazy from the rest of the project: importing this
module is cheap, while constructing ESM2SequenceEncoder loads the selected ESM2
model through fair-esm.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)

ESM2_MODEL_SPECS: Dict[int, Tuple[str, int]] = {
    1280: ("esm2_t33_650M_UR50D", 33),
    2560: ("esm2_t36_3B_UR50D", 36),
    5120: ("esm2_t48_15B_UR50D", 48),
}


class ESM2SequenceEncoder:
    """
    Encode the current protein sequence into per-residue ESM2 embeddings.

    Parameters
    ----------
    embedding_dim:
        Output representation dimension. Supported values are 1280, 2560 and
        5120, matching common ESM2 checkpoints.
    device:
        Torch device for the ESM2 model. "auto" selects CUDA when available.
    mutable_only:
        If true, encode env.current_sequence(mutable_only=True). This is the
        default because the DDQN action layout is aligned to mutable positions:
        one residue row corresponds to 20 amino-acid actions.
    model_dir:
        Optional directory containing local fair-esm checkpoints such as
        esm2_t33_650M_UR50D.pt. When supplied, the encoder loads from disk and
        avoids torch hub downloads.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 1280,
        device: str = "auto",
        mutable_only: bool = True,
        model_dir: str | Path | None = None,
    ) -> None:
        if int(embedding_dim) not in ESM2_MODEL_SPECS:
            raise ValueError(
                "embedding_dim must be one of "
                f"{tuple(ESM2_MODEL_SPECS)}."
            )
        try:
            import esm
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise ImportError(
                "fair-esm is required for ESM2SequenceEncoder. "
                "Install it with: python -m pip install fair-esm"
            ) from exc

        model_name, representation_layer = ESM2_MODEL_SPECS[int(embedding_dim)]
        model_loader: Callable[[], Tuple[Any, Any]] = getattr(esm.pretrained, model_name)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        LOGGER.info(
            "Loading ESM2 model name=%s embedding_dim=%s representation_layer=%s device=%s model_dir=%s",
            model_name,
            embedding_dim,
            representation_layer,
            device,
            model_dir,
        )
        if model_dir is None:
            self.model, alphabet = model_loader()
        else:
            model_path = Path(model_dir).expanduser().resolve() / f"{model_name}.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"ESM2 local checkpoint not found: {model_path}. "
                    "Either provide a directory containing the selected model "
                    "or omit model_dir to use fair-esm's default loader."
                )
            LOGGER.info("Loading ESM2 local checkpoint path=%s", model_path)
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([argparse.Namespace])
            self.model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        self.model.eval()
        self.model.to(torch.device(device))

        self.batch_converter = alphabet.get_batch_converter()
        self.embedding_dim = int(embedding_dim)
        self.representation_layer = int(representation_layer)
        self.device = torch.device(device)
        self.mutable_only = bool(mutable_only)
        self.model_dir = None if model_dir is None else str(Path(model_dir).expanduser().resolve())
        LOGGER.info(
            "ESM2 encoder ready model=%s device=%s output=per_residue",
            model_name,
            self.device,
        )

    def __call__(self, pose: Any, env: Any) -> np.ndarray:
        del pose
        sequence = str(env.current_sequence(mutable_only=self.mutable_only))
        return self.encode_sequence(sequence)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode one sequence while preserving the historical callable API."""

        return self.encode_sequences([sequence])[0]

    def encode_sequences(self, sequences: Sequence[str]) -> list[np.ndarray]:
        """Encode a padded, variable-length sequence batch in one model call."""

        normalized = [str(sequence).strip().upper() for sequence in sequences]
        if not normalized:
            raise ValueError("sequences must contain at least one protein sequence.")
        empty_indices = [index for index, sequence in enumerate(normalized) if not sequence]
        if empty_indices:
            raise ValueError(
                "Cannot encode empty protein sequences at batch indices "
                f"{empty_indices}."
            )

        records = [
            (f"protein_{index}", sequence)
            for index, sequence in enumerate(normalized)
        ]
        _, _, tokens = self.batch_converter(records)
        tokens = tokens.to(self.device)

        LOGGER.debug(
            "Encoding ESM2 batch batch_size=%s min_length=%s max_length=%s "
            "embedding_dim=%s device=%s output=per_residue",
            len(normalized),
            min(map(len, normalized)),
            max(map(len, normalized)),
            self.embedding_dim,
            self.device,
        )
        with torch.inference_mode():
            outputs = self.model(tokens, repr_layers=[self.representation_layer])
            representations = outputs["representations"][self.representation_layer]

        encoded: list[np.ndarray] = []
        for index, sequence in enumerate(normalized):
            # Token layout is BOS, residues..., EOS, then optional padding.
            embedding = representations[index, 1 : len(sequence) + 1]
            array = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
            expected_shape = (len(sequence), self.embedding_dim)
            if array.shape != expected_shape:
                raise RuntimeError(
                    "ESM2 per-residue embedding has shape "
                    f"{array.shape}, expected {expected_shape}."
                )
            encoded.append(array)
        return encoded

"""
ESM2 sequence encoder for MechanicalProteinEnv observations.

The encoder is intentionally lazy from the rest of the project: importing this
module is cheap, while constructing ESM2SequenceEncoder loads the selected ESM2
model through fair-esm.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

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
    Encode the current protein sequence into a fixed-size ESM2 embedding.

    Parameters
    ----------
    embedding_dim:
        Output representation dimension. Supported values are 1280, 2560 and
        5120, matching common ESM2 checkpoints.
    device:
        Torch device for the ESM2 model. "auto" selects CUDA when available.
    pool:
        "mean" averages residue token embeddings, excluding BOS/EOS tokens.
        "cls" returns the BOS token representation.
    mutable_only:
        If true, encode only env.current_sequence(mutable_only=True). The
        default encodes the complete sequence so the embedding keeps global
        protein context.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 1280,
        device: str = "auto",
        pool: str = "mean",
        mutable_only: bool = False,
    ) -> None:
        if int(embedding_dim) not in ESM2_MODEL_SPECS:
            raise ValueError(
                "embedding_dim must be one of "
                f"{tuple(ESM2_MODEL_SPECS)}."
            )
        if pool not in {"mean", "cls"}:
            raise ValueError("pool must be 'mean' or 'cls'.")

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
            "Loading ESM2 model name=%s embedding_dim=%s representation_layer=%s device=%s",
            model_name,
            embedding_dim,
            representation_layer,
            device,
        )
        self.model, alphabet = model_loader()
        self.model.eval()
        self.model.to(torch.device(device))

        self.batch_converter = alphabet.get_batch_converter()
        self.embedding_dim = int(embedding_dim)
        self.representation_layer = int(representation_layer)
        self.device = torch.device(device)
        self.pool = pool
        self.mutable_only = bool(mutable_only)
        LOGGER.info("ESM2 encoder ready model=%s device=%s pool=%s", model_name, self.device, self.pool)

    def __call__(self, pose: Any, env: Any) -> np.ndarray:
        del pose
        sequence = str(env.current_sequence(mutable_only=self.mutable_only))
        if not sequence:
            raise ValueError("Cannot encode an empty protein sequence.")

        _, _, tokens = self.batch_converter([("protein", sequence)])
        tokens = tokens.to(self.device)

        LOGGER.info(
            "Encoding sequence with ESM2 length=%s embedding_dim=%s device=%s pool=%s",
            len(sequence),
            self.embedding_dim,
            self.device,
            self.pool,
        )
        with torch.no_grad():
            outputs = self.model(tokens, repr_layers=[self.representation_layer])
            representations = outputs["representations"][self.representation_layer]

        if self.pool == "cls":
            embedding = representations[0, 0]
        else:
            # Token layout is BOS, residues..., EOS. Average only residue tokens.
            embedding = representations[0, 1 : len(sequence) + 1].mean(dim=0)

        array = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.shape != (self.embedding_dim,):
            raise RuntimeError(
                f"ESM2 embedding has shape {array.shape}, expected {(self.embedding_dim,)}."
            )
        return array

"""
ddqn_agent.py

A standard Double DQN agent for discrete protein-mutation actions.

Implemented features
--------------------
- online Q network and target Q network;
- epsilon-greedy exploration;
- action masking for invalid residue-mutation actions;
- Double DQN target construction;
- Smooth L1 / Huber loss;
- gradient accumulation through micro-batches;
- gradient-norm clipping;
- hard and soft target-network updates;
- optional CUDA automatic mixed precision;
- checkpoint save and restore.

The module is independent of PyRosetta. It consumes ReplayBatch-like objects
whose arrays follow the interface implemented by replay_buffer_module.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)
SUPPORTED_PROTEIN_EMBEDDING_DIMS = (1280, 2560, 5120)
AMINO_ACID_ACTION_DIM = 20


@dataclass(frozen=True)
class DDQNConfig:
    """Hyperparameters for DDQN training."""

    hidden_dims: Tuple[int, ...] = (256, 256)
    embedding_dim: int = 1280
    gamma: float = 0.99
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # GPU-memory control:
    # effective_batch_size = micro_batch_size * gradient_accumulation_steps
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 4

    replay_warmup_size: int = 1_000
    target_sync_interval: int = 250
    max_grad_norm: Optional[float] = 10.0
    huber_beta: float = 1.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    device: str = "auto"
    use_amp: bool = False
    seed: Optional[int] = None

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if not self.hidden_dims or any(int(value) <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if int(self.embedding_dim) not in SUPPORTED_PROTEIN_EMBEDDING_DIMS:
            raise ValueError(
                "embedding_dim must be one of "
                f"{SUPPORTED_PROTEIN_EMBEDDING_DIMS}."
            )
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be within [0, 1].")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if self.replay_warmup_size < 0:
            raise ValueError("replay_warmup_size must be >= 0.")
        if self.target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be > 0.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 or None.")
        if self.huber_beta <= 0:
            raise ValueError("huber_beta must be > 0.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= "
                "epsilon_start <= 1."
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be > 0.")


@dataclass(frozen=True)
class OptimizationResult:
    """Metrics returned after one optimizer step."""

    loss: float
    mean_q_value: float
    mean_target_q_value: float
    mean_absolute_td_error: float
    grad_norm: float
    effective_batch_size: int
    micro_batches: int
    optimization_step: int
    target_synced: bool
    epsilon: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QNetwork(nn.Module):
    """
    Q head for protein-language-model encoded observations.

    Preferred ESM2 input shape is ``(L, embedding_dim)``. In that mode the
    network applies the same residue-wise head to every residue and returns a
    flattened ``(batch, L * 20)`` tensor, matching the environment action layout.

    Legacy pooled or one-hot observations are still supported through the
    original flatten -> projection -> MLP path.
    """

    def __init__(
        self,
        state_shape: Sequence[int],
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        embedding_dim: int = 1280,
    ) -> None:
        super().__init__()

        self.state_shape = _validate_state_shape(state_shape)
        self.action_dim = _validate_positive_int(action_dim, "action_dim")
        self.embedding_dim = _validate_embedding_dim(embedding_dim)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")

        is_per_residue_shape = _is_per_residue_state_shape(
            self.state_shape,
            self.embedding_dim,
        )
        if (
            is_per_residue_shape
            and self.action_dim != self.state_shape[0] * AMINO_ACID_ACTION_DIM
        ):
            raise ValueError(
                "Per-residue QNetwork states require action_dim == residues * 20; "
                f"got action_dim={self.action_dim}, residues={self.state_shape[0]}."
            )
        self.per_residue_mode = is_per_residue_shape

        if self.per_residue_mode:
            dimensions = (self.embedding_dim, *hidden_dims, AMINO_ACID_ACTION_DIM)
            layers = []
            for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
                layers.append(nn.Linear(input_dim, output_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
            self.residue_head = nn.Sequential(*layers)
            self.input_projection = nn.Identity()
            self.network = nn.Identity()
            return

        flattened_dim = int(np.prod(self.state_shape))
        if flattened_dim == self.embedding_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(flattened_dim, self.embedding_dim)

        dimensions = (self.embedding_dim, *hidden_dims, self.action_dim)

        layers = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim == len(self.state_shape):
            states = states.unsqueeze(0)

        if self.per_residue_mode:
            if states.ndim != 3:
                raise ValueError(
                    "Per-residue states must have shape (batch, residues, "
                    f"{self.embedding_dim}), got {tuple(states.shape)}."
                )
            if states.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"Expected per-residue embedding dimension {self.embedding_dim}, "
                    f"got {states.shape[-1]}."
                )
            batch_size, residue_count, _ = states.shape
            residue_q_values = self.residue_head(states)
            return residue_q_values.reshape(
                batch_size,
                residue_count * AMINO_ACID_ACTION_DIM,
            )

        expected_ndim = len(self.state_shape) + 1
        if states.ndim != expected_ndim:
            raise ValueError(
                f"states must have {expected_ndim} dimensions including batch, "
                f"got shape {tuple(states.shape)}."
            )

        if tuple(states.shape[1:]) != self.state_shape:
            raise ValueError(
                f"Expected trailing state shape {self.state_shape}, "
                f"got {tuple(states.shape[1:])}."
            )

        flattened = states.reshape(states.shape[0], -1)
        embedding = self.input_projection(flattened)
        return self.network(embedding)


class DDQNAgent:
    """
    Double DQN agent with invalid-action masks and gradient accumulation.

    Gradient accumulation
    ---------------------
    ``optimize_batch`` receives an effective batch and splits it into
    ``config.micro_batch_size`` chunks. Each micro-batch loss is scaled by the
    total effective-batch size before ``backward()``. Gradients accumulate until
    all chunks are processed. The optimizer then performs exactly one update.

    This lowers peak GPU memory usage while preserving the gradient of the mean
    effective-batch loss, including when the last chunk is smaller.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        state_shape: Sequence[int],
        action_dim: int,
        *,
        config: DDQNConfig = DDQNConfig(),
        online_network: Optional[nn.Module] = None,
        target_network: Optional[nn.Module] = None,
    ) -> None:
        started = time.perf_counter()
        LOGGER.info(
            "Initializing DDQNAgent state_shape=%s action_dim=%s config=%s",
            tuple(state_shape),
            action_dim,
            asdict(config),
        )
        config.validate()

        self.state_shape = _validate_state_shape(state_shape)
        self.action_dim = _validate_positive_int(action_dim, "action_dim")
        self.config = config
        self.per_residue_mode = _is_per_residue_state_shape(
            self.state_shape,
            config.embedding_dim,
        )
        if self.per_residue_mode and self.action_dim != self.state_shape[0] * AMINO_ACID_ACTION_DIM:
            raise ValueError(
                "Per-residue DDQN states require action_dim == residues * 20; "
                f"got action_dim={self.action_dim}, residues={self.state_shape[0]}."
            )
        self.device = self._resolve_device(config.device)
        LOGGER.info("DDQNAgent resolved device=%s", self.device)
        self._rng = np.random.default_rng(config.seed)

        self.online_network = (
            QNetwork(
                self.state_shape,
                self.action_dim,
                hidden_dims=config.hidden_dims,
                embedding_dim=config.embedding_dim,
            )
            if online_network is None
            else online_network
        ).to(self.device)

        self.target_network = (
            copy.deepcopy(self.online_network)
            if target_network is None
            else target_network
        ).to(self.device)

        self._validate_network_output(self.online_network, "online_network")
        self._validate_network_output(self.target_network, "target_network")

        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.amp_enabled = bool(config.use_amp and self.device.type == "cuda")
        self._grad_scaler = self._create_grad_scaler(self.amp_enabled)

        self.environment_steps = 0
        self.optimization_steps = 0
        LOGGER.info(
            "DDQNAgent ready device=%s amp_enabled=%s effective_batch_size=%s elapsed_sec=%.3f",
            self.device,
            self.amp_enabled,
            self.config.effective_batch_size,
            time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return float(
            self.config.epsilon_start
            + progress * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def select_action(
        self,
        state: Any,
        *,
        action_mask: Optional[Any] = None,
        evaluate: bool = False,
        advance_step: bool = True,
    ) -> int:
        """
        Select one valid action.

        Training mode uses epsilon-greedy selection. Evaluation mode always uses
        masked greedy selection and does not advance the epsilon schedule.
        """

        state_array = self._coerce_single_state(state)
        current_action_dim = self._action_dim_for_state_array(state_array)
        mask_array = self._coerce_single_mask(
            action_mask,
            action_dim=current_action_dim,
        )
        valid_actions = np.flatnonzero(mask_array)

        if valid_actions.size == 0:
            raise ValueError("No valid action remains for the current state.")

        use_random_action = (
            not evaluate
            and self._rng.random() < self.epsilon
        )

        if use_random_action:
            action = int(self._rng.choice(valid_actions))
            selection_mode = "random"
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state_array,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                q_values = self.online_network(state_tensor)
                mask_tensor = torch.as_tensor(
                    mask_array,
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0)
                masked_q_values = self.mask_invalid_q_values(q_values, mask_tensor)
                action = int(masked_q_values.argmax(dim=1).item())
            selection_mode = "greedy"

        if not evaluate and advance_step:
            self.environment_steps += 1

        LOGGER.info(
            "DDQNAgent selected action=%s mode=%s evaluate=%s epsilon=%.6f "
            "valid_actions=%s environment_steps=%s",
            action,
            selection_mode,
            evaluate,
            self.epsilon,
            valid_actions.size,
            self.environment_steps,
        )
        return action

    @staticmethod
    def mask_invalid_q_values(q_values: Tensor, action_masks: Tensor) -> Tensor:
        """Replace invalid-action Q values with negative infinity."""

        if q_values.ndim != 2:
            raise ValueError("q_values must have shape (batch_size, action_dim).")
        if action_masks.shape != q_values.shape:
            raise ValueError(
                "action_masks must have the same shape as q_values: "
                f"{tuple(action_masks.shape)} vs {tuple(q_values.shape)}."
            )
        if action_masks.dtype is not torch.bool:
            action_masks = action_masks.to(dtype=torch.bool)

        if not torch.all(action_masks.any(dim=1)):
            raise ValueError("Each masked Q-value row must contain a valid action.")

        return q_values.masked_fill(~action_masks, -torch.inf)

    # ------------------------------------------------------------------
    # DDQN target calculation
    # ------------------------------------------------------------------

    def compute_ddqn_targets(
        self,
        *,
        next_states: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_action_masks: Optional[Tensor],
    ) -> Tensor:
        """
        Calculate Double DQN TD targets.

        The online network chooses the next action. The target network evaluates
        the selected action. Terminal rows may contain all-False masks because
        their bootstrap term is zero.
        """

        with torch.no_grad():
            next_q_online = self.online_network(next_states)

            if next_action_masks is None:
                next_action_masks = torch.ones_like(next_q_online, dtype=torch.bool)
            else:
                next_action_masks = next_action_masks.to(
                    device=next_q_online.device,
                    dtype=torch.bool,
                )

            if next_action_masks.shape != next_q_online.shape:
                raise ValueError(
                    "next_action_masks must match Q-value shape: "
                    f"{tuple(next_action_masks.shape)} vs "
                    f"{tuple(next_q_online.shape)}."
                )

            has_valid_action = next_action_masks.any(dim=1)
            non_terminal = ~dones.to(dtype=torch.bool)

            if torch.any(non_terminal & ~has_valid_action):
                raise ValueError(
                    "A non-terminal next state has no valid action. "
                    "Check environment termination and action-mask logic."
                )

            safe_masks = next_action_masks.clone()
            safe_masks[~has_valid_action] = True

            masked_next_q_online = next_q_online.masked_fill(
                ~safe_masks,
                -torch.inf,
            )
            next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_network(next_states)
            selected_next_q_target = next_q_target.gather(
                dim=1,
                index=next_actions,
            ).squeeze(1)

            return rewards + (
                self.config.gamma
                * (~dones.to(dtype=torch.bool)).to(dtype=rewards.dtype)
                * selected_next_q_target
            )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize_from_replay_buffer(self, replay_buffer: Any) -> Optional[OptimizationResult]:
        """
        Sample an effective batch and perform one optimizer update.

        Returns None until replay warmup and effective-batch requirements are met.
        """

        required_size = max(
            self.config.replay_warmup_size,
            self.config.effective_batch_size,
        )
        if len(replay_buffer) < required_size:
            LOGGER.info(
                "DDQN optimization waiting for replay warmup buffer_size=%s required_size=%s",
                len(replay_buffer),
                required_size,
            )
            return None

        LOGGER.info(
            "DDQN sampling replay batch batch_size=%s buffer_size=%s",
            self.config.effective_batch_size,
            len(replay_buffer),
        )
        batch = replay_buffer.sample(self.config.effective_batch_size)
        return self.optimize_batch(batch)

    def optimize_batch(self, batch: Any) -> OptimizationResult:
        """
        Optimize the online network using micro-batch gradient accumulation.
        """

        optimize_started = time.perf_counter()
        arrays = self._coerce_replay_batch(batch)
        total_size = int(arrays["actions"].shape[0])
        LOGGER.info(
            "DDQN optimize_batch started total_size=%s micro_batch_size=%s device=%s amp=%s",
            total_size,
            self.config.micro_batch_size,
            self.device,
            self.amp_enabled,
        )

        self.online_network.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss_sum = 0.0
        total_q_sum = 0.0
        total_target_sum = 0.0
        total_absolute_td_error_sum = 0.0
        micro_batches = 0

        for start in range(0, total_size, self.config.micro_batch_size):
            stop = min(start + self.config.micro_batch_size, total_size)
            micro_batches += 1
            micro_started = time.perf_counter()
            LOGGER.debug(
                "DDQN micro-batch started index=%s start=%s stop=%s",
                micro_batches,
                start,
                stop,
            )

            states = self._tensor(arrays["states"][start:stop], torch.float32)
            actions = self._tensor(arrays["actions"][start:stop], torch.long)
            rewards = self._tensor(arrays["rewards"][start:stop], torch.float32)
            next_states = self._tensor(arrays["next_states"][start:stop], torch.float32)
            dones = self._tensor(arrays["dones"][start:stop], torch.bool)

            next_action_masks_array = arrays["next_action_masks"]
            next_action_masks = (
                None
                if next_action_masks_array is None
                else self._tensor(
                    next_action_masks_array[start:stop],
                    torch.bool,
                )
            )

            with self._autocast_context():
                td_targets = self.compute_ddqn_targets(
                    next_states=next_states,
                    rewards=rewards,
                    dones=dones,
                    next_action_masks=next_action_masks,
                )

                q_values = self.online_network(states)
                selected_q_values = q_values.gather(
                    dim=1,
                    index=actions.unsqueeze(1),
                ).squeeze(1)

                # Sum reduction followed by division by total_size makes
                # micro-batch accumulation equal to a mean loss over the full
                # effective batch, even for an uneven final chunk.
                loss_sum = F.smooth_l1_loss(
                    selected_q_values,
                    td_targets,
                    beta=self.config.huber_beta,
                    reduction="sum",
                )
                scaled_loss = loss_sum / total_size

            if self.amp_enabled:
                self._grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            td_errors = td_targets.detach() - selected_q_values.detach()
            total_loss_sum += float(loss_sum.detach().item())
            total_q_sum += float(selected_q_values.detach().sum().item())
            total_target_sum += float(td_targets.detach().sum().item())
            total_absolute_td_error_sum += float(td_errors.abs().sum().item())
            LOGGER.debug(
                "DDQN micro-batch complete index=%s rows=%s loss_sum=%.8f elapsed_sec=%.3f",
                micro_batches,
                stop - start,
                float(loss_sum.detach().item()),
                time.perf_counter() - micro_started,
            )

        if self.amp_enabled:
            self._grad_scaler.unscale_(self.optimizer)

        if self.config.max_grad_norm is None:
            grad_norm = self._calculate_grad_norm()
        else:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                max_norm=self.config.max_grad_norm,
            )
            grad_norm = float(grad_norm_tensor.detach().item())

        if not np.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            LOGGER.error("DDQN gradient norm is non-finite grad_norm=%s", grad_norm)
            raise FloatingPointError(
                "Gradient norm is NaN or infinity. "
                "Inspect rewards, Q values and learning rate."
            )

        if self.amp_enabled:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()
        LOGGER.info("DDQN optimizer step applied grad_norm=%.8f", grad_norm)

        self.optimization_steps += 1

        target_synced = False
        if self.optimization_steps % self.config.target_sync_interval == 0:
            self.hard_sync_target_network()
            target_synced = True
            LOGGER.info("DDQN hard target sync triggered optimization_step=%s", self.optimization_steps)

        result = OptimizationResult(
            loss=total_loss_sum / total_size,
            mean_q_value=total_q_sum / total_size,
            mean_target_q_value=total_target_sum / total_size,
            mean_absolute_td_error=total_absolute_td_error_sum / total_size,
            grad_norm=grad_norm,
            effective_batch_size=total_size,
            micro_batches=micro_batches,
            optimization_step=self.optimization_steps,
            target_synced=target_synced,
            epsilon=self.epsilon,
        )
        LOGGER.info(
            "DDQN optimize_batch complete optimization_step=%s loss=%.8f "
            "mean_q=%.8f mean_target=%.8f mean_abs_td=%.8f micro_batches=%s "
            "target_synced=%s elapsed_sec=%.3f",
            result.optimization_step,
            result.loss,
            result.mean_q_value,
            result.mean_target_q_value,
            result.mean_absolute_td_error,
            result.micro_batches,
            result.target_synced,
            time.perf_counter() - optimize_started,
        )
        return result

    # ------------------------------------------------------------------
    # Target-network lifecycle
    # ------------------------------------------------------------------

    def hard_sync_target_network(self) -> None:
        """Copy all online-network weights into the target network."""

        LOGGER.info("DDQN hard target sync started")
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        LOGGER.info("DDQN hard target sync complete")

    def soft_sync_target_network(self, tau: float) -> None:
        """Polyak update: target <- tau * online + (1 - tau) * target."""

        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be within (0, 1].")

        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_network.parameters(),
                self.online_network.parameters(),
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(online_parameter, alpha=tau)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a complete training checkpoint payload."""

        payload: Dict[str, Any] = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "state_shape": self.state_shape,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "optimization_steps": self.optimization_steps,
            "numpy_rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore training state into a compatible agent instance."""

        if int(payload.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported DDQN checkpoint version.")
        if tuple(payload["state_shape"]) != self.state_shape:
            raise ValueError("Checkpoint state_shape is incompatible with this agent.")
        if int(payload["action_dim"]) != self.action_dim:
            raise ValueError("Checkpoint action_dim is incompatible with this agent.")

        self.online_network.load_state_dict(payload["online_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.target_network.eval()
        self.optimizer.load_state_dict(payload["optimizer"])
        self._move_optimizer_state_to_device()

        self.environment_steps = int(payload["environment_steps"])
        self.optimization_steps = int(payload["optimization_steps"])
        self._rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])

        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

    def save_checkpoint(self, path: PathLike) -> None:
        """Save model, optimizer and scheduling state."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("DDQN save_checkpoint started path=%s", path)
        torch.save(self.state_dict(), path)
        LOGGER.info("DDQN save_checkpoint complete path=%s", path)

    @classmethod
    def from_checkpoint(
        cls,
        path: PathLike,
        *,
        map_location: Optional[Union[str, torch.device]] = None,
        config_override: Optional[DDQNConfig] = None,
    ) -> "DDQNAgent":
        """Construct and restore an agent from a checkpoint file."""

        payload = _torch_load(path, map_location=map_location)
        config = (
            DDQNConfig(**payload["config"])
            if config_override is None
            else config_override
        )

        agent = cls(
            state_shape=tuple(payload["state_shape"]),
            action_dim=int(payload["action_dim"]),
            config=config,
        )
        agent.load_state_dict(payload)
        return agent

    # ------------------------------------------------------------------
    # Validation and utilities
    # ------------------------------------------------------------------

    def _coerce_replay_batch(self, batch: Any) -> Dict[str, Optional[np.ndarray]]:
        required_names = (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
        )
        missing = [name for name in required_names if not hasattr(batch, name)]
        if missing:
            raise ValueError(f"Replay batch is missing fields: {missing}.")

        states = np.asarray(batch.states, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.int64)
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        next_states = np.asarray(batch.next_states, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.bool_)

        if self.per_residue_mode:
            if states.ndim != 3 or states.shape[-1] != self.config.embedding_dim:
                raise ValueError(
                    "Per-residue batch states must have shape "
                    f"(batch, residues, {self.config.embedding_dim}), got {states.shape}."
                )
        else:
            if states.ndim != len(self.state_shape) + 1:
                raise ValueError("Batch states have an invalid number of dimensions.")
            if tuple(states.shape[1:]) != self.state_shape:
                raise ValueError(
                    f"Batch states must have trailing shape {self.state_shape}, "
                    f"got {tuple(states.shape[1:])}."
                )

        batch_size = int(states.shape[0])
        if batch_size <= 0:
            raise ValueError("Replay batch must contain at least one transition.")
        if next_states.shape != states.shape:
            raise ValueError("next_states must have the same shape as states.")
        batch_action_dim = self._action_dim_for_state_batch(states)
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape (batch_size,).")
        if rewards.shape != (batch_size,):
            raise ValueError("rewards must have shape (batch_size,).")
        if dones.shape != (batch_size,):
            raise ValueError("dones must have shape (batch_size,).")
        if np.any(actions < 0) or np.any(actions >= batch_action_dim):
            raise ValueError("Replay batch contains an out-of-range action.")
        if not np.all(np.isfinite(states)):
            raise ValueError("states contain NaN or infinity.")
        if not np.all(np.isfinite(next_states)):
            raise ValueError("next_states contain NaN or infinity.")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("rewards contain NaN or infinity.")

        next_action_masks_value = getattr(batch, "next_action_masks", None)
        next_action_masks: Optional[np.ndarray]
        if next_action_masks_value is None:
            next_action_masks = None
        else:
            next_action_masks = np.asarray(next_action_masks_value, dtype=np.bool_)
            if next_action_masks.shape != (batch_size, batch_action_dim):
                raise ValueError(
                    "next_action_masks must have shape "
                    f"{(batch_size, batch_action_dim)}."
                )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "next_action_masks": next_action_masks,
        }

    def _coerce_single_state(self, state: Any) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32)
        if self.per_residue_mode:
            if array.ndim != 2 or array.shape[-1] != self.config.embedding_dim:
                raise ValueError(
                    "per-residue state must have shape "
                    f"(residues, {self.config.embedding_dim}), got {array.shape}."
                )
        elif array.shape != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}, got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("state contains NaN or infinity.")
        return array

    def _coerce_single_mask(
        self,
        action_mask: Optional[Any],
        *,
        action_dim: Optional[int] = None,
    ) -> np.ndarray:
        expected_action_dim = self.action_dim if action_dim is None else int(action_dim)
        if action_mask is None:
            return np.ones(expected_action_dim, dtype=np.bool_)
        array = np.asarray(action_mask, dtype=np.bool_)
        if array.shape != (expected_action_dim,):
            raise ValueError(
                f"action_mask must have shape {(expected_action_dim,)}, "
                f"got {array.shape}."
            )
        return array

    def _action_dim_for_state_array(self, state: np.ndarray) -> int:
        if self.per_residue_mode:
            return int(state.shape[0]) * AMINO_ACID_ACTION_DIM
        return self.action_dim

    def _action_dim_for_state_batch(self, states: np.ndarray) -> int:
        if self.per_residue_mode:
            if states.ndim != 3 or states.shape[-1] != self.config.embedding_dim:
                raise ValueError(
                    "Per-residue batch states must have shape "
                    f"(batch, residues, {self.config.embedding_dim}), got {states.shape}."
                )
            return int(states.shape[1]) * AMINO_ACID_ACTION_DIM
        return self.action_dim

    def _tensor(self, array: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _calculate_grad_norm(self) -> float:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.online_network.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.stack(norms).norm(2).item())

    def _validate_network_output(self, network: nn.Module, name: str) -> None:
        with torch.no_grad():
            dummy = torch.zeros(
                (1, *self.state_shape),
                dtype=torch.float32,
                device=self.device,
            )
            output = network(dummy)
        if output.shape != (1, self.action_dim):
            raise ValueError(
                f"{name} must output shape {(1, self.action_dim)} for one "
                f"state, got {tuple(output.shape)}."
            )

    @staticmethod
    def _resolve_device(requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return device

    @staticmethod
    def _create_grad_scaler(enabled: bool):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(self.device)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _validate_state_shape(state_shape: Sequence[int]) -> Tuple[int, ...]:
    shape = tuple(int(value) for value in state_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("state_shape must contain positive dimensions.")
    return shape


def _validate_embedding_dim(value: Any) -> int:
    embedding_dim = int(value)
    if embedding_dim not in SUPPORTED_PROTEIN_EMBEDDING_DIMS:
        raise ValueError(
            "embedding_dim must be one of "
            f"{SUPPORTED_PROTEIN_EMBEDDING_DIMS}."
        )
    return embedding_dim


def _is_per_residue_state_shape(
    state_shape: Sequence[int],
    embedding_dim: int,
) -> bool:
    shape = tuple(int(value) for value in state_shape)
    return len(shape) == 2 and shape[-1] == int(embedding_dim)


def _torch_load(
    path: PathLike,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Mapping[str, Any]:
    try:
        return torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(Path(path), map_location=map_location)

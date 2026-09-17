from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import run_rl
from model.agent_module.ddqn_agent import QNetwork


class _FakeEncoder:
    def __init__(self, embedding_dim: int = 1280, **_: object) -> None:
        self.embedding_dim = int(embedding_dim)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        return np.zeros((len(sequence), self.embedding_dim), dtype=np.float32)


class _PositionPriorityNetwork(torch.nn.Module):
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = states.shape
        values = torch.arange(
            length * run_rl.AMINO_ACID_ACTION_DIM,
            dtype=torch.float32,
            device=states.device,
        )
        return values.expand(batch_size, -1)


def _fake_policy(*, feature_dim: int = 1280) -> run_rl.LoadedPolicy:
    return run_rl.LoadedPolicy(
        network=_PositionPriorityNetwork(),  # type: ignore[arg-type]
        checkpoint_path=Path("fake.pt"),
        embedding_dim=1280,
        state_feature_dim=feature_dim,
        device=torch.device("cpu"),
    )


def _pdb_atom(serial: int, residue: str, chain: str, residue_number: int) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} {chain:1s}{residue_number:4d}    "
        "  10.000  10.000  10.000  1.00 20.00           C\n"
    )


def test_parse_single_record_fasta(tmp_path: Path) -> None:
    path = tmp_path / "1ACF.fasta"
    path.write_text(">protein A\nACD\nEFG\n", encoding="utf-8")

    protein = run_rl.parse_fasta_input(path)

    assert protein.name == "1ACF"
    assert protein.sequence == "ACDEFG"
    assert "protein A" in protein.source_detail


def test_parse_fasta_rejects_multiple_records(tmp_path: Path) -> None:
    path = tmp_path / "two.fasta"
    path.write_text(">one\nACD\n>two\nEFG\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        run_rl.parse_fasta_input(path)


def test_parse_pdb_extracts_one_canonical_chain(tmp_path: Path) -> None:
    path = tmp_path / "1ACF.pdb"
    path.write_text(
        _pdb_atom(1, "ALA", "A", 1)
        + _pdb_atom(2, "CYS", "A", 2)
        + "HETATM    3  C1  LIG A   3      10.000  10.000  10.000  1.00 20.00           C\n",
        encoding="utf-8",
    )

    protein = run_rl.parse_pdb_input(path)

    assert protein.name == "1ACF"
    assert protein.sequence == "AC"
    assert "chain='A'" in protein.source_detail


def test_parse_pdb_rejects_multiple_protein_chains(tmp_path: Path) -> None:
    path = tmp_path / "multi.pdb"
    path.write_text(
        _pdb_atom(1, "ALA", "A", 1) + _pdb_atom(2, "GLY", "B", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one canonical protein chain"):
        run_rl.parse_pdb_input(path)


def test_parse_update_counts_supports_defaults_lists_and_ranges() -> None:
    assert run_rl.parse_update_counts(None, 26) == (24, 25, 26)
    assert run_rl.parse_update_counts("3", 10) == (3,)
    assert run_rl.parse_update_counts("2,4:6,4", 10) == (2, 4, 5, 6)
    assert run_rl.parse_update_counts("11", 10) == (10,)
    assert run_rl.parse_update_counts("2,11:12", 10) == (2, 10)
    assert run_rl.parse_update_counts(None, 10) == (10,)


def test_action_mask_blocks_noops_and_visited_positions() -> None:
    mask = run_rl.build_action_mask("AC", {1})

    assert mask.shape == (40,)
    assert not mask[0]
    assert mask[1:20].all()
    assert not mask[20:].any()


def test_design_sequence_mutates_unique_positions_and_returns_snapshots() -> None:
    snapshots, mutations = run_rl.design_sequence(
        "ACD",
        update_counts=(1, 3),
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        policy=_fake_policy(feature_dim=1281),
    )

    assert set(snapshots) == {1, 3}
    assert sum(left != right for left, right in zip("ACD", snapshots[1])) == 1
    assert sum(left != right for left, right in zip("ACD", snapshots[3])) == 3
    assert len({mutation.position for mutation in mutations}) == 3


def test_load_policy_infers_per_residue_checkpoint_shape(tmp_path: Path) -> None:
    network = QNetwork(
        state_shape=(1, 1281),
        action_dim=20,
        hidden_dims=(8,),
        embedding_dim=1280,
    )
    checkpoint = tmp_path / "agent_00000001.pt"
    torch.save(
        {
            "state_shape": (1, 1281),
            "action_dim": 20,
            "config": {"embedding_dim": 1280, "hidden_dims": (8,)},
            "online_network": network.state_dict(),
            "environment_steps": 12,
            "optimization_steps": 3,
        },
        checkpoint,
    )

    loaded = run_rl.load_policy(tmp_path, device=torch.device("cpu"))

    assert loaded.embedding_dim == 1280
    assert loaded.state_feature_dim == 1281
    assert loaded.includes_visited_mask
    assert loaded.checkpoint_path == checkpoint


def test_cli_designs_folder_and_uses_required_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "1ACF.fasta").write_text(">1ACF\nACD\n", encoding="utf-8")
    monkeypatch.setattr(run_rl, "load_policy", lambda *args, **kwargs: _fake_policy())
    monkeypatch.setattr(run_rl, "ESM2SequenceEncoder", _FakeEncoder)

    exit_code = run_rl.main(
        [
            "--model-path",
            str(tmp_path / "unused.pt"),
            "--input-type",
            "sequence",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--updates",
            "2",
        ]
    )

    assert exit_code == 0
    output = output_dir / "1ACF_2.fasta"
    assert output.is_file()
    lines = output.read_text(encoding="ascii").splitlines()
    assert lines[0] == ">1ACF_2"
    assert len(lines[1]) == 3
    assert sum(left != right for left, right in zip("ACD", lines[1])) == 2

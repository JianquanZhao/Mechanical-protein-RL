from pathlib import Path

import numpy as np
import pandas as pd

from sft_data_generation.hashing import task_id
from sft_data_generation.manifest import assert_no_cluster_leakage, parse_single_chain_sequence
from sft_data_generation.proposals import build_candidate_manifest
from sft_data_generation.scan import shard_for_task
from sft_data_generation.search import select_global_beam
from sft_data_generation.statistics import aggregate_replicates
from sft_data_generation.tasks import build_action_tasks, decode_action, encode_action


def _pdb_line(serial: int, atom: str, residue: str, chain: str, residue_number: int) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{residue_number:4d}    "
        "   0.000   0.000   0.000  1.00 20.00           C  \n"
    )


def test_parse_single_chain_sequence(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    lines = []
    serial = 1
    for residue_number, residue in enumerate(("ALA", "CYS", "ASP"), start=1):
        for atom in ("N", "CA", "C", "O"):
            lines.append(_pdb_line(serial, atom, residue, "A", residue_number))
            serial += 1
    pdb.write_text("".join(lines), encoding="utf-8")
    chain, sequence = parse_single_chain_sequence(pdb)
    assert chain == "A"
    assert sequence == "ACD"


def test_action_contract_and_task_ids_are_deterministic() -> None:
    for position in (1, 7):
        for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
            assert decode_action(encode_action(position, amino_acid)) == (position, amino_acid)
    states = pd.DataFrame(
        [
            {
                "state_id": "s1",
                "root_state_id": "s1",
                "parent_state_id": "",
                "protein_id": "p1",
                "split": "train",
                "depth": 0,
                "pdb_path": "/tmp/p1.pdb",
                "sequence": "AC",
                "visited_positions": "",
            }
        ]
    )
    first = build_action_tasks(
        states, repeat_seeds=[11, 23], reward_contract_hash="contract", split_names=["train"]
    )
    second = build_action_tasks(
        states, repeat_seeds=[11, 23], reward_contract_hash="contract", split_names=["train"]
    )
    assert len(first) == 2 * 19 * 2
    assert first["task_id"].tolist() == second["task_id"].tolist()
    assert [shard_for_task(value, 8) for value in first["task_id"]] == [
        shard_for_task(value, 8) for value in second["task_id"]
    ]


def test_aggregate_replicates_and_global_beam() -> None:
    rows = []
    for action, values in {1: [2.0, 2.5, 1.5], 2: [-1.0, -0.5, -1.5], 3: [0.1, -0.1, 0.0]}.items():
        for repeat, value in enumerate(values):
            rows.append(
                {
                    "schema_version": "1.0",
                    "task_id": f"t{action}_{repeat}",
                    "state_id": "s1",
                    "root_state_id": "root",
                    "parent_state_id": "",
                    "protein_id": "p1",
                    "split": "train",
                    "depth": 0,
                    "pose_path": "/tmp/p1.pdb",
                    "root_pdb_path": "/tmp/p1.pdb",
                    "sequence": "AC",
                    "candidate_sequence": "DC",
                    "visited_positions": "",
                    "action_index": action,
                    "position": 1,
                    "wild_type_amino_acid": "A",
                    "mutant_amino_acid": "D",
                    "proposal_sources": "full_scan",
                    "reward_contract_hash": "contract",
                    "success": True,
                    "failure_reason": "",
                    "marginal_reward_scaled": value,
                    "delta_strength_marginal": value,
                    "delta_toughness_marginal": value,
                }
            )
    labels = aggregate_replicates(
        pd.DataFrame(rows), bootstrap_samples=500, positive_noise_boundary=0.0
    )
    classes = labels.set_index("action_index")["label_class"].to_dict()
    assert classes[1] == "confirmed_positive"
    assert classes[2] == "confirmed_negative"
    frontier = select_global_beam(labels, beam_width=2)
    assert len(frontier) == 2
    assert frontier.iloc[0]["action_index"] == 1


def test_cluster_leakage_is_rejected() -> None:
    frame = pd.DataFrame(
        {"cluster_id": ["c1", "c1"], "split": ["train", "test"]}
    )
    try:
        assert_no_cluster_leakage(frame)
    except ValueError as exc:
        assert "cross" in str(exc)
    else:
        raise AssertionError("Expected cluster leakage validation to fail")


def test_candidate_merge_keeps_budget_and_random_coverage() -> None:
    states = pd.DataFrame(
        [{"state_id": "s1", "sequence": "ACD", "visited_positions": ""}]
    )
    proposals = pd.DataFrame(
        [
            {"state_id": "s1", "action_index": 1, "proposal_source": "sft", "proposal_score": 2.0},
            {"state_id": "s1", "action_index": 2, "proposal_source": "esm2", "proposal_score": 1.0},
        ]
    )
    candidates = build_candidate_manifest(
        states, budget=8, random_count=3, proposal_tables=[proposals]
    )
    assert len(candidates) == 8
    assert candidates["action_index"].nunique() == 8
    assert (candidates["proposal_sources"] == "random").sum() >= 3

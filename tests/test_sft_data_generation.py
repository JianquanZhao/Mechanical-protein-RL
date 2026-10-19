from pathlib import Path

import numpy as np
import pandas as pd

from sft_data_generation.hashing import task_id
from sft_data_generation.embeddings import cache_state_embeddings
from sft_data_generation.manifest import assert_no_cluster_leakage, parse_single_chain_sequence
from sft_data_generation.proposals import build_candidate_manifest
from sft_data_generation.release import create_compact_release
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


def test_compact_release_nested_split_dead_zone_and_embedding_cache(tmp_path: Path) -> None:
    states = []
    actions = []
    for index in range(8):
        state_id = f"s{index}"
        states.append(
            {
                "state_id": state_id,
                "protein_id": f"p{index}",
                "cluster_id": f"c{index}",
                "split": "train",
                "sequence": "AC",
                "visited_positions": "",
            }
        )
        for action_index, reward in ((1, 0.005), (2, 0.2), (3, -0.2)):
            actions.append(
                {
                    "state_id": state_id,
                    "action_index": action_index,
                    "reward_mean": reward,
                    "reward_lcb": reward,
                    "reward_ucb": reward,
                    "confidence_weight": 1.0,
                    "label_class": "confirmed_positive" if reward > 0 else "confirmed_negative",
                    "delta_strength_mean": reward / 2,
                    "delta_toughness_mean": reward / 4,
                }
            )
    states.append(
        {
            "state_id": "outer_validation",
            "protein_id": "held_out",
            "cluster_id": "held_out_cluster",
            "split": "validation",
            "sequence": "AC",
            "visited_positions": "",
        }
    )
    states_path = tmp_path / "split_manifest.csv"
    actions_path = tmp_path / "labels.csv"
    pd.DataFrame(states).to_csv(states_path, index=False)
    pd.DataFrame(actions).sort_values(["state_id", "action_index"]).to_csv(actions_path, index=False)
    release_dir = tmp_path / "release"
    summary = create_compact_release(
        states_path=states_path,
        actions_path=actions_path,
        output_dir=release_dir,
        metadata={"dataset_version": "test"},
        split_fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        split_seed=7,
        dead_zone_tau=0.01,
        uncertain_weight=0.25,
        chunk_size=5,
    )
    released = pd.read_csv(release_dir / "states.csv", keep_default_na=False)
    assert set(released["split"]) == {"train", "validation", "test"}
    assert set(released["manifest_split"]) == {"train"}
    assert "outer_validation" not in set(released["state_id"])
    assert_no_cluster_leakage(released)
    assert summary["label_counts"] == {
        "confirmed_positive": 8,
        "confirmed_negative": 8,
        "near_zero_or_uncertain": 8,
    }

    class EncoderStub:
        embedding_dim = 4

        @staticmethod
        def encode_sequences(sequences):
            return [np.ones((len(sequence), 4), dtype=np.float32) for sequence in sequences]

    embedding_summary = cache_state_embeddings(
        release_dir,
        embedding_dim=4,
        dtype="float16",
        batch_size=3,
        encoder=EncoderStub(),
    )
    assert embedding_summary["complete"] is True
    cached = pd.read_csv(release_dir / "states.csv", keep_default_na=False)
    assert cached["embedding_path"].astype(str).ne("").all()
    assert all((release_dir / value).is_file() for value in cached["embedding_path"])

from pathlib import Path

from model.dataset_module.dataset import ProteinStructureDataset, discover_structure_files


def test_discover_structure_files_finds_pdbs_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.pdb").write_text("A", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.ent").write_text("B", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    files = discover_structure_files(tmp_path)

    assert [path.name for path in files] == ["a.pdb", "b.ent"]


def test_dataset_split_writes_and_reads_indices(tmp_path: Path) -> None:
    for name in ("a.pdb", "b.pdb", "c.pdb", "d.pdb"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    dataset = ProteinStructureDataset.from_folder(
        tmp_path,
        val_fraction=0.25,
        seed=3,
        recreate_indices=True,
    )

    assert dataset.train_index_path.is_file()
    assert dataset.val_index_path.is_file()
    assert len(dataset.train_paths) == 3
    assert len(dataset.val_paths) == 1

    restored = ProteinStructureDataset.from_folder(tmp_path)
    assert restored.train_paths == dataset.train_paths
    assert restored.val_paths == dataset.val_paths

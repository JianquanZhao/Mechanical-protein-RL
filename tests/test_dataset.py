from pathlib import Path

from model.dataset_module.dataset import (
    ProteinStructureDataset,
    count_canonical_protein_residues,
    discover_structure_files,
    filter_protein_structure_files,
    is_protein_structure_file,
)


def write_pdb(path: Path, residues: list[tuple[str, int]]) -> None:
    lines = []
    atom_index = 1
    for residue_name, residue_number in residues:
        lines.append(
            f"ATOM  {atom_index:5d}  CA  {residue_name:>3s} A{residue_number:4d}"
            "      11.104  13.207   9.447  1.00 20.00           C\n"
        )
        atom_index += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


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
        write_pdb(tmp_path / name, [("ALA", 1), ("GLY", 2)])

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


def test_protein_preprocessing_filters_dna_structures(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    dna = tmp_path / "dna.pdb"
    mixed = tmp_path / "mixed.pdb"
    write_pdb(protein, [("ALA", 1), ("GLY", 2)])
    write_pdb(dna, [(" DA", 1), (" DC", 2)])
    write_pdb(mixed, [(" DG", 1), ("SER", 2)])

    assert count_canonical_protein_residues(protein) == 2
    assert count_canonical_protein_residues(dna) == 0
    assert is_protein_structure_file(protein)
    assert not is_protein_structure_file(dna)

    valid = filter_protein_structure_files([protein, dna, mixed])
    assert [path.name for path in valid] == ["mixed.pdb", "protein.pdb"]


def test_dataset_split_uses_only_preprocessed_protein_files(tmp_path: Path) -> None:
    write_pdb(tmp_path / "protein_a.pdb", [("ALA", 1), ("GLY", 2)])
    write_pdb(tmp_path / "protein_b.pdb", [("SER", 1), ("THR", 2)])
    write_pdb(tmp_path / "dna_only.pdb", [(" DA", 1), (" DT", 2)])

    dataset = ProteinStructureDataset.from_folder(
        tmp_path,
        val_fraction=0.5,
        seed=1,
        recreate_indices=True,
    )

    indexed_names = {path.name for path in (*dataset.train_paths, *dataset.val_paths)}
    assert indexed_names == {"protein_a.pdb", "protein_b.pdb"}
    assert "dna_only.pdb" not in dataset.train_index_path.read_text(encoding="utf-8")
    assert "dna_only.pdb" not in dataset.val_index_path.read_text(encoding="utf-8")


def test_existing_indices_are_preprocessed_and_rewritten(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    dna = tmp_path / "dna.pdb"
    write_pdb(protein, [("ALA", 1), ("GLY", 2)])
    write_pdb(dna, [(" DA", 1), (" DT", 2)])
    (tmp_path / "train_index.txt").write_text("protein.pdb\ndna.pdb\n", encoding="utf-8")
    (tmp_path / "val_index.txt").write_text("", encoding="utf-8")

    dataset = ProteinStructureDataset.from_folder(tmp_path)

    assert dataset.train_paths == (protein.resolve(),)
    assert dataset.val_paths == tuple()
    assert (tmp_path / "train_index.txt").read_text(encoding="utf-8") == "protein.pdb\n"

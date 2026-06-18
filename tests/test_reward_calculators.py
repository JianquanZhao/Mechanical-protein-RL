from types import SimpleNamespace

import pytest

from model.reward_module.reward_calculators import StepRewardCalculator


class FakeResidue:
    def __init__(self, name3: str, atoms: dict[str, tuple[float, float, float]]) -> None:
        self._name3 = name3
        self._atoms = {name.upper(): coords for name, coords in atoms.items()}

    def name3(self) -> str:
        return self._name3

    def has(self, atom_name: str) -> bool:
        return atom_name.upper() in self._atoms

    def xyz(self, atom_name: str):
        x, y, z = self._atoms[atom_name.upper()]
        return SimpleNamespace(x=x, y=y, z=z)


class FakePose:
    def __init__(self, residues: list[FakeResidue]) -> None:
        self._residues = residues

    def residue(self, index: int) -> FakeResidue:
        return self._residues[index - 1]


def make_calculator(*, policy: str = "skip_residue") -> StepRewardCalculator:
    calculator = StepRewardCalculator.__new__(StepRewardCalculator)
    calculator.rmsd_atom_names = ("N", "CA", "C", "O")
    calculator.rmsd_missing_atom_policy = policy
    calculator.rmsd_missing_penalty = 5.0
    calculator.min_rmsd_atoms = 3
    return calculator


def residue(offset: float = 0.0, *, missing_atoms: tuple[str, ...] = tuple()) -> FakeResidue:
    atoms = {
        "N": (0.0 + offset, 0.0, 0.0),
        "CA": (1.0 + offset, 0.0, 0.0),
        "C": (1.0 + offset, 1.0, 0.0),
        "O": (0.0 + offset, 1.0, 0.0),
    }
    for atom_name in missing_atoms:
        atoms.pop(atom_name, None)
    return FakeResidue("ALA", atoms)


def test_local_rmsd_skips_residue_with_missing_atoms() -> None:
    calculator = make_calculator(policy="skip_residue")
    reference_pose = FakePose([residue(), residue()])
    mobile_pose = FakePose([residue(), residue(missing_atoms=("N",))])

    rmsd, status, atom_count, skipped = calculator._local_rmsd(
        reference_pose,
        mobile_pose,
        (1, 2),
    )

    assert rmsd == pytest.approx(0.0)
    assert status == "skipped_missing_atoms"
    assert atom_count == 4
    assert skipped == (2,)


def test_local_rmsd_penalizes_when_too_few_atoms_remain() -> None:
    calculator = make_calculator(policy="penalize")
    reference_pose = FakePose([residue()])
    mobile_pose = FakePose([residue(missing_atoms=("N",))])

    rmsd, status, atom_count, skipped = calculator._local_rmsd(
        reference_pose,
        mobile_pose,
        (1,),
    )

    assert rmsd == pytest.approx(5.0)
    assert status == "penalized_missing_atoms"
    assert atom_count == 0
    assert skipped == (1,)

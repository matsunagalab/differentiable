"""Bit-parity regression tests for the `create_lig.cc` Python port.

Strategy:
 1. Parse 1KXQ's `.out` file.
 2. Run `generate_lig_coords` on the raw ligand PDB for pose indices 1, 50, 100.
 3. Compare atom-by-atom against the C++-generated `complex.{1,50,100}.pdb`
    ligand section (atoms 3909..4824 of the concatenated file).

The C++ tool writes coordinates with `%8.3f`, so the ground-truth granularity
is 1e-3 Å. We allow atol=1.1e-3 to absorb rounding of the last digit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.io import parse_pdb_ms
from zdock.zdock_output import generate_lig_coords, parse_out_file


_BENCH_1KXQ = (
    Path(__file__).resolve().parent.parent.parent
    / "docking" / "protein" / "1KXQ"
)

# The receptor has 3908 atoms; everything after is the ligand.
_N_REC = 3908


@pytest.fixture(scope="module")
def zdock_out():
    p = _BENCH_1KXQ / "1KXQ.zd3.0.2.fg.fixed.out"
    if not p.exists():
        pytest.skip(f"missing fixture: {p}")
    return parse_out_file(p)


@pytest.fixture(scope="module")
def lig_raw() -> np.ndarray:
    p = _BENCH_1KXQ / "1KXQ_l_u.pdb.ms"
    if not p.exists():
        pytest.skip(f"missing fixture: {p}")
    return parse_pdb_ms(p).xyz


def _reference_lig_pose(pose_idx: int) -> np.ndarray:
    """Read the ligand section of complex.<pose_idx>.pdb, which is the
    receptor concatenated with a fresh ligand PDB from create_lig."""
    p = _BENCH_1KXQ / f"complex.{pose_idx}.pdb"
    if not p.exists():
        pytest.skip(f"missing fixture: {p}")
    atoms = parse_pdb_ms(p)
    assert len(atoms) >= _N_REC, f"expected at least {_N_REC} atoms in {p}"
    return atoms.xyz[_N_REC:]


def test_out_file_header(zdock_out):
    """Sanity-check parsed header fields."""
    assert zdock_out.N == 128
    assert zdock_out.spacing == pytest.approx(1.2)
    assert zdock_out.pose_rot.shape == (54000, 3)
    assert zdock_out.pose_trans.shape == (54000, 3)
    # First pose's ZDOCK score field.
    assert zdock_out.pose_score[0] == pytest.approx(1716.525, abs=1e-3)


@pytest.mark.parametrize("pose_idx", [1, 50, 100])
def test_create_lig_parity(zdock_out, lig_raw, pose_idx):
    """The Python port must reproduce the C++ `create_lig` output to 1e-3 Å."""
    expected = _reference_lig_pose(pose_idx)  # (N_lig, 3) float (3-dp)

    lig_xyz = torch.as_tensor(lig_raw, dtype=torch.float64)
    all_poses = generate_lig_coords(
        lig_xyz, zdock_out, n_poses=pose_idx, dtype=torch.float64,
    )
    # pose index in the .out file is 1-based (matches pred_num in create.pl).
    got = all_poses[pose_idx - 1].numpy()
    assert got.shape == expected.shape

    # The ground truth is rounded to 3 decimal places by %8.3f, so expected
    # can be off from the true float by up to 0.5e-3 per axis. Allow 1.1e-3.
    max_err = np.abs(got - expected).max()
    np.testing.assert_allclose(
        got, expected, atol=1.1e-3,
        err_msg=f"pose {pose_idx}: max |delta| = {max_err:.2e} Å",
    )


def test_generate_lig_coords_shape(zdock_out, lig_raw):
    """Shape/device/dtype contract: full 54000-pose regen produces (F, N, 3)."""
    lig_xyz = torch.as_tensor(lig_raw, dtype=torch.float64)
    out = generate_lig_coords(lig_xyz, zdock_out, dtype=torch.float64)
    assert out.shape == (54000, lig_raw.shape[0], 3)
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


def test_switched_ligand_not_supported(tmp_path):
    """Header with a switch_num should raise NotImplementedError until
    createPDBrev is ported."""
    bad = tmp_path / "switched.out"
    bad.write_text(
        "128\t1.2\t1\n"
        "0.0 0.0 0.0\n"
        "0.0 0.0 0.0\n"
        "rec.pdb 0 0 0\n"
        "lig.pdb 0 0 0\n"
    )
    with pytest.raises(NotImplementedError, match="switched-ligand"):
        parse_out_file(bad)

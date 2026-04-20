"""Tests for `zdock.data.load_training_dataset`.

Uses a tiny synthetic h5 built inside a tmp_path so the test is independent
of any external dataset.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from zdock.data import list_proteins, load_training_dataset


def _write_fixture(path: Path, *, with_hit_mask: bool, with_rmsd: bool):
    """Build a two-protein synthetic h5 with minimal shapes."""
    rng = np.random.default_rng(0)
    N_rec, N_lig, F = 4, 3, 5
    with h5py.File(path, "w") as f:
        f.attrs["rmsd_threshold_angstrom"] = 2.5
        f.attrs["zdock_benchmark"] = "synthetic"
        for name, seed in [("PRT1", 1), ("PRT2", 2)]:
            g = f.create_group(name)
            g["rec_xyz"] = rng.standard_normal((N_rec, 3)).astype("float32")
            g["rec_radius"] = np.ones(N_rec, dtype="float32")
            g["rec_sasa"] = np.ones(N_rec, dtype="float32")
            g["rec_atomtype_id"] = np.arange(1, N_rec + 1, dtype="int64")
            g["rec_charge_id"] = np.ones(N_rec, dtype="int64")
            g["lig_xyz"] = rng.standard_normal((F, N_lig, 3)).astype("float32")
            g["lig_radius"] = np.ones(N_lig, dtype="float32")
            g["lig_sasa"] = np.ones(N_lig, dtype="float32")
            g["lig_atomtype_id"] = np.arange(1, N_lig + 1, dtype="int64")
            g["lig_charge_id"] = np.ones(N_lig, dtype="int64")
            if with_rmsd:
                # Deterministic RMSD sequence per-protein: [1.0, 2.0, 3.0, 4.0, 5.0].
                # At threshold 2.5, hit_mask should be [T, T, F, F, F] → 2 hits.
                g["rmsd"] = np.arange(1.0, F + 1.0, dtype="float32") + (seed - 1)
            if with_hit_mask:
                # Write an intentionally different mask so we can tell it apart
                # from the rmsd-derived one.
                mask = np.array([True, False, True, False, False])
                g["hit_mask"] = mask


def test_load_with_hit_mask_only(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=False)
    proteins = load_training_dataset(p, dtype=torch.float64)
    assert len(proteins) == 2
    for prot in proteins:
        assert prot.rec_xyz.shape == (4, 3)
        assert prot.lig_xyz.shape == (5, 3, 3)
        assert prot.hit_mask.dtype == torch.bool
        assert prot.hit_mask.tolist() == [True, False, True, False, False]
        assert prot.rmsd is None


def test_load_with_rmsd_derives_mask_from_default_threshold(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=False, with_rmsd=True)
    proteins = load_training_dataset(p, dtype=torch.float64)
    # PRT1 rmsd = [1, 2, 3, 4, 5]; threshold 2.5 → [T, T, F, F, F]
    assert proteins[0].hit_mask.tolist() == [True, True, False, False, False]
    # PRT2 rmsd = [2, 3, 4, 5, 6]; threshold 2.5 → [T, F, F, F, F]
    assert proteins[1].hit_mask.tolist() == [True, False, False, False, False]


def test_rmsd_override_threshold(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=True)
    # Stored hit_mask is [T, F, T, F, F]; override recomputes from rmsd.
    proteins = load_training_dataset(p, rmsd_threshold_angstrom=4.5)
    # PRT1 rmsd = [1, 2, 3, 4, 5] → [T, T, T, T, F]
    assert proteins[0].hit_mask.tolist() == [True, True, True, True, False]


def test_override_without_rmsd_raises(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=False)
    with pytest.raises(ValueError, match="rmsd dataset missing"):
        load_training_dataset(p, rmsd_threshold_angstrom=4.0)


def test_list_proteins_returns_sorted_groups(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=False)
    assert list_proteins(p) == ["PRT1", "PRT2"]


def test_protein_names_whitelist(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=False)
    # Load in the caller-specified order; skip PRT1.
    proteins = load_training_dataset(p, protein_names=["PRT2"])
    assert len(proteins) == 1


def test_missing_protein_raises(tmp_path):
    p = tmp_path / "fixture.h5"
    _write_fixture(p, with_hit_mask=True, with_rmsd=False)
    with pytest.raises(KeyError, match="NOPE"):
        load_training_dataset(p, protein_names=["NOPE"])

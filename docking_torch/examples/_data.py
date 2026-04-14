"""Shared data loader for the `examples/` scripts.

The three example scripts import only from this module for dataset
plumbing, so future dataset expansion (FOLLOWUPS F-2 / F-3) happens in
one place: extend `DATASETS` (and optionally `DEFAULT_TRAIN_SET` /
`DEFAULT_TEST_SET`) and the scripts pick it up.

Current state: only 1KXQ has a Julia-generated HDF5 reference. When
F-2 lands (`phase5_1F51.h5`, `phase5_2VDB.h5`), add entries here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.train import ProteinInputs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS: dict[str, Path] = {
    "1KXQ": REPO_ROOT / "docking" / "tests" / "refs" / "1KXQ" / "phase5_scores.h5",
    # F-2 (FOLLOWUPS.md): extend Julia generate_refs.jl for 1F51, 2VDB and
    # drop the HDF5 under docking/tests/refs/<ID>/, then uncomment:
    # "1F51": REPO_ROOT / "docking" / "tests" / "refs" / "1F51" / "phase5_scores.h5",
    # "2VDB": REPO_ROOT / "docking" / "tests" / "refs" / "2VDB" / "phase5_scores.h5",
}

DEFAULT_TRAIN_SET: list[str] = ["1KXQ"]
DEFAULT_TEST_SET: list[str] = []


def _load_h5(path: Path) -> dict:
    out: dict = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = f[key][()]
    return out


def _2d(arr) -> np.ndarray:
    a = np.asarray(arr)
    return a.T if a.ndim == 2 and a.shape[0] == 3 else a


def _3d(arr) -> np.ndarray:
    a = np.asarray(arr)
    return a.transpose(2, 1, 0) if a.ndim == 3 and a.shape[0] == 3 else a


def _resolve(protein_id: str) -> Path:
    if protein_id not in DATASETS:
        known = ", ".join(sorted(DATASETS)) or "(none)"
        raise KeyError(
            f"Unknown protein id {protein_id!r}. "
            f"Known: {known}. Add it to DATASETS in examples/_data.py."
        )
    path = DATASETS[protein_id]
    if not path.exists():
        raise FileNotFoundError(
            f"Reference HDF5 for {protein_id!r} not found at {path}. "
            f"Regenerate it with `cd ../docking && julia tests/julia_ref/generate_refs.jl`."
        )
    return path


def load_raw_h5(protein_id: str) -> dict:
    """Return the raw numpy contents of the protein's phase5 HDF5."""
    return _load_h5(_resolve(protein_id))


@dataclass
class LoadedProtein:
    """Bundle returned by load_protein(): the ProteinInputs plus raw HDF5
    scalars/arrays that scripts sometimes want to cross-reference (e.g.
    the Julia-computed `score_coulomb_total` used by the visualization
    demo as a sanity-check column)."""
    id: str
    inputs: ProteinInputs
    raw: dict


def load_protein(
    protein_id: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    hit_mask: torch.Tensor | None = None,
    n_hit_default: int = 3,
) -> LoadedProtein:
    """Load a protein from its phase5 HDF5 into a `ProteinInputs`.

    `hit_mask=None` synthesizes a mask with the first `n_hit_default`
    poses marked Hit (the same convention as tests/test_phase7_train.py
    for the 1KXQ smoke test; real RMSD-based labels will be added once
    F-2 reference data includes them).
    """
    raw = load_raw_h5(protein_id)
    device = torch.device(device)

    def T(key: str, *, int_: bool = False) -> torch.Tensor:
        arr = np.asarray(raw[key])
        if arr.ndim == 3:
            arr = _3d(arr)
        elif arr.ndim == 2:
            arr = _2d(arr)
        target_dtype = torch.int64 if int_ else dtype
        t = torch.as_tensor(arr)
        if t.is_floating_point():
            t = t.to(target_dtype)
        else:
            t = t.to(target_dtype)
        return t.to(device)

    lig_xyz = T("lig_xyz")
    F = lig_xyz.shape[0]
    if hit_mask is None:
        hit_mask = torch.zeros(F, dtype=torch.bool, device=device)
        hit_mask[: min(n_hit_default, F)] = True
    else:
        hit_mask = hit_mask.to(device=device, dtype=torch.bool)
        if hit_mask.shape != (F,):
            raise ValueError(
                f"hit_mask shape {tuple(hit_mask.shape)} must be ({F},) to match lig_xyz"
            )

    inputs = ProteinInputs(
        rec_xyz=T("rec_xyz"),
        rec_radius=T("rec_radius"),
        rec_sasa=T("rec_sasa"),
        rec_atomtype_id=T("rec_atomtype_id", int_=True),
        rec_charge_id=T("rec_charge_id", int_=True),
        lig_xyz=lig_xyz,
        lig_radius=T("lig_radius"),
        lig_sasa=T("lig_sasa"),
        lig_atomtype_id=T("lig_atomtype_id", int_=True),
        lig_charge_id=T("lig_charge_id", int_=True),
        hit_mask=hit_mask,
    )
    return LoadedProtein(id=protein_id, inputs=inputs, raw=raw)


def resolve_device(name: str) -> torch.device:
    """Map "auto" / "cpu" / "cuda" / "mps" to a concrete torch.device."""
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_dtype(device: torch.device) -> torch.dtype:
    """Same convention as tests/conftest.py: float64 on CPU, float32 elsewhere."""
    return torch.float64 if device.type == "cpu" else torch.float32

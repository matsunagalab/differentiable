# zdock — Differentiable ZDOCK in PyTorch

PyTorch port of the Julia implementation in `../docking/` (`docking.jl` +
`train_param-apart.ipynb`). Lets you **optimize the 157 ZDOCK scoring
parameters (α, β, IFACE 144, charge 11) end-to-end with Adam**. Nine bugs
found in the Julia reference are fixed here; see `../docking/PORT_PLAN.md`
for the full list.

## Highlights

- **Pure PyTorch** — NumPy and h5py are only for reference I/O
- **Same code runs on macOS (MPS) and Linux (CUDA)** out of the box
- Numerical agreement with the Julia reference is guaranteed by pytest
  (all 28 tests green on CPU + MPS; training and physics tests included)
- PyTorch autograd replaces the buggy hand-written `rrule` in Julia
- Per-frame work is fused into a single einsum for GPU-friendly batching
- **Physically correct Coulombic electrostatics** by default
  (Chen 2002/2003 formulation), with a `legacy` mode for bit-exact
  thesis reproduction — see below

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd docking_torch

# Create a venv, resolve from uv.lock, install `zdock` in editable mode
uv sync

# Run the tests on CPU / float64 (~18 s)
uv run pytest -q

# Apple Silicon — use the Metal backend (~30 s)
ZDOCK_DEVICE=mps uv run pytest -q

# Linux with an NVIDIA GPU — use CUDA
ZDOCK_DEVICE=cuda uv run pytest -q
```

`uv sync` picks the right PyTorch wheel per platform (see below).

## Per-platform PyTorch selection

The routing is declared in `pyproject.toml` under `[tool.uv.sources]`:

| OS | PyTorch wheel | Compute backend | Notes |
|---|---|---|---|
| macOS (Intel / Apple Silicon) | `torch` from PyPI | Metal / MPS | `ZDOCK_DEVICE=mps` on Apple Silicon |
| Linux | `torch+cu124` from pytorch.org | CUDA 12.4 | `ZDOCK_DEVICE=cuda` |
| Windows | PyPI (CPU) | CPU | See below to opt into CUDA |

### Choosing a different CUDA version (cu121, cu126, …)

Edit the index block in `pyproject.toml` and re-run `uv sync`:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"           # rename to cu121 / cu126 etc.
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

### CPU-only Linux (no GPU)

Comment out the `torch` entry under `[tool.uv.sources]` and `uv sync`,
or for a one-off install:

```bash
uv sync --index https://pypi.org/simple --reinstall-package torch
```

### Windows + CUDA

Widen the marker in `[tool.uv.sources]`:

```toml
torch = [
    { index = "pytorch-cu124",
      marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

## macOS OpenMP / threading

`tests/conftest.py` sets the following env vars automatically:

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
```

Without them, the macOS combination of Python 3.14 + PyTorch 2.11 + h5py
segfaults on non-trivial tensor ops due to multiple libomp copies in the
same process. Linux is unaffected. If you launch workers in subprocesses
and want parallel CPU math, unset these in the child.

## What you can do

### 1. Forward scoring

```python
import torch
from zdock.score import docking_score_elec
from zdock.atomtypes import iface_ij, charge_score

device = "mps"          # or "cuda", "cpu"
dtype  = torch.float32

scores = docking_score_elec(
    rec_xyz, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id,
    lig_xyz,            # (F, N_lig, 3) — multiple poses in one call
    lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id,
    alpha=torch.tensor(0.01, device=device, dtype=dtype),
    iface_ij_flat=iface_ij(device=device, dtype=dtype, flat=True),
    beta=torch.tensor(3.0, device=device, dtype=dtype),
    charge_score=charge_score(device=device, dtype=dtype),
    elec_mode="coulomb",   # default; "legacy" for thesis reproduction
)
# scores: (F,) docking score per pose
```

F is the number of poses evaluated together; the frame dimension is
batched through einsum.

#### `elec_mode` — electrostatics model selection

- **`"coulomb"`** (default, physically correct per Chen 2002 / 2003):
  Receptor generates a Coulombic potential `V(r) = Σⱼ qⱼ / |r − rⱼ|`
  (zeroed inside the receptor SC shape), ligand stores its partial
  charge at the nearest grid cell of each atom, and `score_elec` accumulates
  the Coulomb interaction energy `ΣΣ qᵢ qⱼ / rᵢⱼ` over all (lig × rec)
  atom pairs. β scales this energy in `score_total`.

- **`"legacy"`**: the original training notebook's formulation —
  `Σq / Σr` pseudo-quantity restricted to same-atomtype pairs, no
  receptor-core zeroing. Preserved for bit-exact reproduction of the
  master's thesis numbers. Fails several physics sanity checks (see
  `tests/test_physics.py`) and is therefore **not** recommended for new
  work.

See `../docking/PORT_PLAN.md` entries B10 / B11 / B12 / B13 for the
details of what was wrong with the original `legacy` formulation.

### 2. Gradients

Just let autograd do it:

```python
alpha = torch.tensor(0.01, requires_grad=True)
beta  = torch.tensor(3.0,  requires_grad=True)
iface = iface_ij(flat=True).clone().requires_grad_(True)
charge = charge_score().clone().requires_grad_(True)

scores = docking_score_elec(..., alpha, iface, beta, charge)
loss = (scores - target).pow(2).sum()
loss.backward()
print(alpha.grad, beta.grad, iface.grad, charge.grad)
```

The Julia hand-written `rrule` had significant bugs (B5/B6/B7 — gradient
8×–23× too large). PyTorch autograd is used instead and has been verified
against Julia central-difference gradients (see `test_phase6_grad.py`).

### 3. Adam training

```python
from zdock.train import ProteinInputs, train

p = ProteinInputs(
    rec_xyz=..., rec_radius=..., rec_sasa=...,
    rec_atomtype_id=..., rec_charge_id=...,
    lig_xyz=..., lig_radius=..., lig_sasa=...,
    lig_atomtype_id=..., lig_charge_id=...,
    hit_mask=...,                 # (F,) bool — Positive vs Negative
)

out = train([p], n_epoch=200, lr=0.01, device="mps", dtype=torch.float32)
print("final loss:", out["history"]["loss"][-1])
print("optimized α, β:", out["alpha"].item(), out["beta"].item())
```

The loss is the B2-fixed 6-term MSE (all terms contribute; the Julia
notebook's newline-continuation bug silently dropped five of them).

### 4. Regenerating the reference HDF5 (usually unnecessary)

The `.h5` files under `../docking/tests/refs/1KXQ/` are the gold outputs
from the patched Julia reference. If you need to regenerate them:

```bash
cd ../docking
julia tests/julia_ref/generate_refs.jl        # Phase 1–5 refs (~30 s)
julia tests/julia_ref/gradcheck_fd_export.jl  # FD gradients for B-6 (~1 min)
```

## Directory layout

```
docking_torch/
├── pyproject.toml          ← uv project manifest (with platform routing)
├── uv.lock                 ← universal lock for darwin + linux
├── .python-version         ← pin Python 3.12
├── README.md               ← this file
├── src/zdock/
│   ├── __init__.py
│   ├── atomtypes.py        ← LUTs (ace_score, iface_ij, charge_score)
│   │                         + set_radius / set_atomtype_id / set_charge
│   ├── _atomtype_rules.py  ← auto-generated: 167 atom-type rules
│   ├── geom.py             ← rotate, golden_section_spiral, generate_grid,
│   │                         decenter, orient (PCA via SVD)
│   ├── sasa.py             ← compute_sasa (neighbor-packed, GPU-friendly)
│   ├── spread.py           ← scatter_add-based spreads (nearest/neighbors,
│   │                         add/substitute, calculate_distance, coulomb)
│   ├── score.py            ← docking_score_elec (frame-batched, 144 →
│   │                         12×12 einsum)
│   └── train.py            ← ProteinInputs, Adam loop, B2-fixed loss
└── tests/
    ├── conftest.py         ← device/dtype/tol fixtures, HDF5 loader
    ├── test_phase1.py      ← LUTs, rotate, spiral, atom types (11 tests)
    ├── test_phase2.py      ← SASA, grid generation (3)
    ├── test_phase3.py      ← spread / calculate_distance (5)
    ├── test_phase5.py      ← docking_score_elec end-to-end (1)
    ├── test_phase6_grad.py ← autograd vs Julia FD (1)
    ├── test_phase7_train.py← Adam smoke (30 ep) + `slow` 200-epoch (1+1)
    ├── test_physics.py     ← Coulombic ELEC primitives: sign, 1/r, superposition,
    │                         cross-type pair contribution, autograd (5)
    └── test_orient.py      ← orient matches Julia SVD bit-exact (1)
```

## Timing on 1KXQ, 10 poses, α=0.01, β=3.0

| Operation | CPU (float64, M2) | MPS (float32, M2) | Notes |
|---|---|---|---|
| Setup (SASA on 3908-atom receptor) | ≈ 2 s | ≈ 0.8 s | neighbor-packed |
| `docking_score_elec` forward / pose | ≈ 10 ms | ≈ 70 ms | MPS launch overhead dominates |
| `docking_score_elec` backward / pose | ≈ 18 ms | ≈ 100 ms | autograd |
| 200-epoch training (1KXQ, 10 poses) | ≈ 13 s | ≈ 40 s | loss drops ~78% |
| Full pytest suite | ≈ 18 s | ≈ 35 s | 24 tests |

MPS gains relative to CPU scale with F. At F=100 on a CUDA A100, expect
50–100× speedup over single-core CPU.

## Troubleshooting

### MPS warning about `aten::linalg_svd`

`orient` uses a 3×3 SVD. MPS lacks that op, but we explicitly run the
3×3 matrix on CPU and copy back, so no warning leaks through.

### `Fatal Python error: Segmentation fault` on macOS

Make sure `OMP_NUM_THREADS=1` is set. `conftest.py` handles this for
pytest; if you run a standalone script:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python my_script.py
```

### Linux sync didn't install the CUDA build

`uv sync` requires network access to `download.pytorch.org`. Confirm
with `nvidia-smi` that CUDA 12.4 drivers are present, then

```bash
uv sync --reinstall-package torch
```

### Out-of-memory during training

Reduce `atom_chunk` in `compute_sasa`:

```python
sasa = compute_sasa(xyz, radius, atom_chunk=32)  # default 16 (CPU) / 512 (GPU)
```

Or split `lig_xyz` along the frame axis and call `docking_score_elec`
multiple times, summing the gradients manually before `.backward()`.

### Autograd disagreement with Julia reference

The reference in `test_phase6_grad.py` only spot-checks (α, β, three
`iface` entries, three `charge` entries). If you modify the forward and
see any of these drift, run `gradcheck_fd_export.jl` on a tiny (2–3
pose) case and compare all 144 + 11 entries — one of the ELEC /
IFACE grouping conventions (see B9 in `PORT_PLAN.md`) is easy to break.

## Pointers

- **Parent project**: `../docking/` — Julia version, master thesis, full plan
- **Port plan**: `../docking/PORT_PLAN.md` — bug list B1–B15 and rationale
- **Follow-up tasks** (not required for basic use):
  [FOLLOWUPS.md](FOLLOWUPS.md) — F-1 CHARMM19 charges, F-2 full 3-protein
  training, F-3 test-protein Rank evaluation, F-4 `torch.compile`.
- **B4 physics note**: `../docking/tests/julia_ref/b4_physics_report.md`
  (Σq/Σr vs Σq/r) — superseded by the Phase C section of PORT_PLAN.md
- **Julia reference tests**: `../docking/tests/julia_ref/README.md`
- **Master's thesis**: `../docking/master_thesis/ICS-25M-23MM336.pdf`

## License

MIT — inherits from the parent repository.

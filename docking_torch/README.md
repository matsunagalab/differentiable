# zdock — Differentiable ZDOCK in PyTorch

PyTorch port of the Julia implementation in `../docking/` (`docking.jl` +
`train_param-apart.ipynb`). Lets you **optimize the 156 learnable ZDOCK
scoring parameters (α, IFACE 144, charge 11) end-to-end with Adam**;
β is held fixed at 3.0 because it is mathematically redundant with an
overall scaling of `charge_score`. Nine bugs found in the Julia
reference are fixed here; see `../docking/PORT_PLAN.md` for the full
list.

## Highlights

- **Pure PyTorch** — NumPy and h5py are only for reference I/O
- **Same code runs on macOS (MPS) and Linux (CUDA)** out of the box
- Numerical agreement with the Julia reference is guaranteed by pytest
  (all 82 tests green on CPU + MPS; training, physics, and FFT search
  tests included)
- **FFT-based docking pose search** (`zdock.search.docking_search`)
  evaluates every translation of the ligand at fixed rotation in one
  batched FFT — the same math as the upstream ZDOCK Fortran binary
  but end-to-end PyTorch / autograd-safe. See `PORT_PLAN_FFT.md` for
  the Julia-reference audit that preceded the port.
- PyTorch autograd replaces the buggy hand-written `rrule` in Julia
- Per-frame work is fused into a single einsum for GPU-friendly batching
- **Physically correct Coulombic electrostatics** by default
  (Chen 2002/2003 formulation), with a `legacy` mode for bit-exact
  thesis reproduction — see below

## How the scoring works

This section describes what `docking_score_elec` computes and why
each piece is differentiable, so you can read `score.py` with a
mental model already in place.

### What we're computing

Protein–protein docking asks: given two protein structures (a
**receptor** and a **ligand**), which spatial arrangements (poses)
represent a plausible binding complex? This package does **not**
search for poses — that is done upstream by ZDOCK, which outputs
hundreds of candidate poses. Instead, `docking_score_elec` **re-scores**
F candidate poses in a single batched call and returns a score per pose:

```
score[f] = α · S_SC[f]  +  S_IFACE[f]  +  β · S_ELEC[f]
```

| Term | Full name | What it captures |
|------|-----------|-----------------|
| S_SC | Shape Complementarity | How well the molecular surfaces fit together |
| S_IFACE | Interface statistics | Atom-type pairwise contact preferences |
| S_ELEC | Electrostatic energy | Coulombic charge–charge interaction |
| α, β | Weight scalars | Relative importance of SC vs ELEC |

**Learnable parameters:** α (1) + iface_ij (12×12 = 144) +
charge_score (11) = **156 total**, all differentiable via PyTorch autograd.
β (also 1 scalar) is a function input but held fixed at 3.0 during
training — it is scale-redundant with `charge_score` (see §
*Differentiability and training*).

### Grid-based evaluation

Atoms are **scattered onto a 3D grid** (default spacing 3 Å) so that
scoring reduces to grid inner products — fast and GPU-friendly.

```
Receptor atoms ──→ scatter ──→ Receptor grids ──┐
     (one-time precompute)     (SC, H[j], V)    │
                                                 │  inner product
Ligand pose[f] ──→ scatter ──→ Ligand grids  ───┤  (per frame f)
     (repeated per pose)       (SC, L[i], Q)    │
                                                 ↓
                                  score[f] = α·SC + IFACE + β·ELEC
```

The receptor grids are built **once**; only the ligand side is
recomputed per pose. Cost is O(V + N_atoms) per pose (V = number of
grid cells), much cheaper than the naive O(N_rec × N_lig) all-pairs
evaluation.

### Shape complementarity (SC)

SC detects whether the molecular **surfaces** fit snugly while
penalizing **core–core** clashes (steric overlap).

1. Atoms are classified as **surface** (SASA > 1.0 Å²) or **core**.
2. For each molecule, atoms are spread onto the grid with radii-scaled
   cutoffs, producing a real part (shape envelope) and an imaginary
   part (penetration penalty weight — 3.5 for surface, 12.25 for core).
3. The receptor's and ligand's grids are combined via **complex
   multiplication** and summed over all cells:
   - Surface–surface contact → **positive** (favorable fit).
   - Core–core overlap → **large negative** (steric clash).

This follows Chen & Weng, *Proteins* 2002.

### Interface statistics (IFACE)

IFACE captures which **atom-type pairs** are in contact at the
receptor–ligand interface.

1. All atoms carry one of **12 atom types** (atomtype_id 1–12, assigned
   by residue name + atom name via `atomtypes.py`).
2. **Receptor** — for each type j, a grid slab `H[j]` counts how many
   type-j atoms lie within 6 Å of each cell → shape (12, nx, ny, nz).
3. **Ligand** — for each type i, `L[f, i]` is a binary indicator of
   whether a type-i atom occupies each cell → shape (F, 12, nx, ny, nz).
4. The pairwise overlap is computed in one shot by **einsum**:
   `T[f, i, j] = Σ_cells L[f, i, cell] × H[j, cell]`.
5. The IFACE score weights each pair by the learnable matrix:
   `S_IFACE[f] = Σ_ij iface_ij[i, j] × T[f, i, j]`.

The 144 entries of `iface_ij` encode atom-type "affinities" — training
adjusts them so that native-like contacts contribute more than
non-native ones.

### Electrostatics (ELEC) — Coulomb mode

ELEC evaluates the **Coulombic interaction energy** between all
receptor and ligand atoms, mediated through the grid.

1. **Receptor** — build a potential grid
   `V(r) = Σ_j q_j / |r − r_j|` using `spread_neighbors_coulomb`
   (cutoff 8 Å). Cells inside the receptor core (SC envelope > 0) are
   zeroed out (Chen 2002, §2.2).
2. **Ligand** — scatter each atom's partial charge q onto the nearest
   grid cell → `Q[f]` of shape (F, nx, ny, nz).
3. **Interaction** — inner product:
   `S_ELEC[f] = Σ_cells V(cell) × Q[f, cell] ≈ ΣΣ q_i q_j / r_ij`.
   - Opposite charges (attraction) → negative.
   - Like charges (repulsion) → positive.
4. β scales the electrostatic contribution in the total score.

The 11-element `charge_score` look-up table maps each atom's charge
type to a partial charge value. These are also learnable — training
can fine-tune the effective charge strengths.

This follows Chen, Li & Bhatt, *Proteins* 2003.

### Differentiability and training

Every operation in the pipeline — `scatter_add`, `einsum`,
element-wise arithmetic, reductions — has a built-in PyTorch backward
pass. Calling `loss.backward()` propagates gradients through the
entire scoring function to all 156 learnable parameters (α, iface_ij,
charge_score). β is held fixed at 3.0: because `score_elec` is linear
(coulomb mode) or quadratic (legacy mode) in `charge_score`, any β can
be absorbed into an overall scaling of `charge_score`, so training β
separately just adds a scale-redundant degree of freedom.

Training (see `train.py`) works as follows:

1. Poses are labelled **Hit** (RMSD ≤ threshold to the native complex)
   or **Miss**.
2. Targets are set once before training: all Hit poses target the
   maximum pre-training score; all Miss poses target the minimum.
3. A per-class MSE loss (Hit term + Miss term, averaged separately)
   is minimized with **Adam** over 200 epochs.
4. After training, Hit poses should score higher → better ranking.

The Julia notebook had a bug (B2) that silently dropped five of the
six MSE terms; the port fixes this, so all protein contributions
actually affect the gradient.

Two training objectives are available via `train(..., loss=...)`:

- **`"split_mse"`** (default): the Hit/Miss per-class MSE described
  above — matches the Julia reference.
- **`"rank"`**: **ListNet** listwise cross-entropy on RMSD —
  `-Σ softmax(-rmsd / T) · log softmax(scores)`. Uses the raw RMSD
  values directly (no Hit/Miss threshold), so continuous ordering is
  preserved. `T` (Å, `listnet_temperature`, default 5.0) controls
  target peakedness. Requires `rmsd` on every `ProteinInputs`.

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
beta  = torch.tensor(3.0)  # held fixed — redundant with charge scaling
iface = iface_ij(flat=True).clone().requires_grad_(True)
charge = charge_score().clone().requires_grad_(True)

scores = docking_score_elec(..., alpha, iface, beta, charge)
loss = (scores - target).pow(2).sum()
loss.backward()
print(alpha.grad, iface.grad, charge.grad)
```

The Julia hand-written `rrule` had significant bugs (B5/B6/B7 — gradient
8×–23× too large). PyTorch autograd is used instead and has been verified
against Julia central-difference gradients (see `test_phase6_grad.py`).

### 3. Adam training

Low-level API — `zdock.train.train(proteins, n_epoch=…, lr=…)` takes a
list of `ProteinInputs` and returns a dict with the learned `alpha`,
`iface`, `charge` (β is kept fixed at 3.0 internally) plus a loss
history:

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
print("optimized α:", out["alpha"].item(), "  (β fixed at 3.0)")

# Rank-based alternative (ListNet on RMSD — requires p.rmsd):
out = train([p], n_epoch=200, lr=0.01, loss="rank", listnet_temperature=5.0)
```

The default loss is the B2-fixed 6-term MSE (all terms contribute;
the Julia notebook's newline-continuation bug silently dropped five
of them). Pass `loss="rank"` for the ListNet objective on RMSD — see
§ *Differentiability and training* above.

For the end-to-end dataset workflow (load BM4 → 70/15/15 split → lr
grid search on val → evaluate test), use `examples/02_train.py` and
`examples/03_evaluate.py` instead — see **Examples → (2) / (3)** below.

### 4. Training on the BM4 dataset (student workflow)

The **ZDOCK Benchmark 4** covers 176 protein-protein pairs × 54000
ZDOCK poses each. `scripts/build_training_dataset.py` consolidates it
into a single HDF5 file (`datasets/bm4_full.h5`, ~96 GB) that
`examples/02_train.py` splits 70 / 15 / 15 (train / val / test), sweeps
a small learning-rate grid, picks the lr that maximises the val
Hit-in-top-K metric, and stores the held-out test names in the ckpt so
`examples/03_evaluate.py` can evaluate the untouched test split later.

#### Step 1. Obtain the dataset

**Option A — download from the shared location** (ask your advisor
for the URL). ~96 GB single file; drop it under
`docking_torch/datasets/bm4_full.h5` and move to step 2.

**Option B — rebuild from raw inputs** (~50 min, needs the sibling
Julia project checked out):

```bash
uv run python scripts/build_training_dataset.py \
    --benchmark-root ../docking/decoys_bm4_zd3.0.2_6deg_fixed \
    --output datasets/bm4_full.h5 \
    --skip-errors
```

This walks every protein in `results/`, reads the extended `*.pdb.ms`
files, reconstructs all 54000 ligand poses per protein via the
Python port of `create_lig.cc`, reads the `*.rmsds` files, and
writes one group per protein with `hit_mask = rmsd <= 2.5 Å`.
47 proteins fail the Julia atomtype rule ladder (cofactors like
HEM / ATP / MSE) and are skipped; the remaining **129 proteins**
include the thesis triple (1KXQ + 1F51 + 2VDB) and the hold-out
pair (1CGI + 1ZHI).

Smaller builds are fine too — `--max-poses 2000` gives a ~3-4 GB
file suitable for email attachment / quick experimentation.

#### Step 2. Train with the one-command example

```bash
uv run python examples/02_train.py
uv run python examples/03_evaluate.py     # evaluates the held-out test split
```

See **Examples → (2) / (3)** below for the knobs. The defaults (10
proteins × top-100 poses × 50 epochs × 3-point lr grid) finish in
under a minute on CPU; scale up with `--n-proteins all --top-k 2000`
once you trust the pipeline.

#### Step 3. (Optional) Load the dataset yourself

If you want to go beyond the example script:

```python
from zdock.data import list_proteins, load_training_dataset

print(list_proteins("datasets/bm4_full.h5"))          # 129 protein IDs

proteins = load_training_dataset(
    "datasets/bm4_full.h5",
    protein_names=["1KXQ", "1F51", "2VDB"],          # pick any subset
    rmsd_threshold_angstrom=5.0,                      # optional: override
    device="mps", dtype=torch.float32,
)
# -> list of ProteinInputs, one per name, ready for train([...])
```

Every group carries the raw `rmsd` array too, so you can re-derive
`hit_mask` with any threshold without rebuilding.

### 5. Regenerating the reference HDF5 (usually unnecessary)

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
│   ├── train.py            ← ProteinInputs, Adam loop, B2-fixed loss
│   ├── data.py             ← list_proteins / load_training_dataset
│   │                         (consolidated BM4 HDF5 reader)
│   ├── io.py               ← PDB / .ms / rmsds readers for dataset build
│   ├── zdock_output.py     ← ZDOCK `.out` + `create_lig` Python port
│   ├── search.py           ← FFT-based docking search (`docking_search`)
│   └── rotation_grid.py    ← random / Euler-grid quaternion samplers
└── tests/                  ← 58 tests: numerical parity with Julia,
                              SASA / SVD / autograd / train / physics
```

## Examples

Three runnable demo scripts live under `examples/`. They cover the
normal workflow: score-and-visualize, train, and before/after eval.
All outputs land in `docking_torch/out/`.

### Setup

```bash
cd docking_torch
uv sync                 # installs zdock editable + matplotlib (dev group)
# or, in an existing venv:
# uv pip install -e ".[plot]"
```

Then any example runs via `uv run python examples/<script>.py`. Pick a
GPU by prefixing `CUDA_VISIBLE_DEVICES=<idx>` and/or passing
`--device cuda` (default `auto`: CUDA → MPS → CPU).

### (1) Scoring + visualization — `examples/01_score_and_visualize.py`

```bash
uv run python examples/01_score_and_visualize.py
# CUDA_VISIBLE_DEVICES=2 uv run python examples/01_score_and_visualize.py --device cuda
```

Loads the 1KXQ phase5 reference (10 ZDOCK candidate poses), scores them
with default parameters (α=0.01, β=3.0, LUT iface/charge), prints a
ranked table alongside Julia's `score_coulomb_total` for sanity, and
writes:

- `out/01_poses.pdb` — multi-model point-cloud (receptor as chain R,
  top-K ligand poses as chain L) openable in PyMOL / ChimeraX
- `out/01_poses.png` — matplotlib 3D scatter preview

### (2) Training on BM4 with val-based lr selection — `examples/02_train.py`

One-command entry point for students. Proteins are split **70 % train
/ 15 % val / 15 % test** at the protein level (deterministic per
`--seed`), the model is trained once per lr in `--lr-grid` on the
train set, and the lr that maximises the **val Hit-in-top-K** metric
is kept. The test split is **not** touched here — its names are
written into the ckpt for `examples/03_evaluate.py` to pick up.

```bash
# Defaults: 10 proteins, top-100 poses, 50 epochs, lr grid {0.003,0.01,0.03}
uv run python examples/02_train.py

# Full dataset (129 proteins × top-2000 poses) on MPS / CUDA
uv run python examples/02_train.py --n-proteins all --top-k 2000 --device mps

# Custom split + lr grid + seed:
uv run python examples/02_train.py \
    --n-proteins 30 --top-k 500 --epochs 100 \
    --train-split 0.7 --val-split 0.15 \
    --lr-grid 0.001,0.003,0.01,0.03 --seed 123

# Rank-based (ListNet on RMSD) instead of Hit/Miss split-MSE:
uv run python examples/02_train.py --loss rank --listnet-temperature 5.0
```

The two knobs students use most:

| Flag | Meaning | Default |
|---|---|---|
| `--n-proteins` | how many proteins from the 129 available | 10 |
| `--top-k` | how many top-ranked ZDOCK poses per protein | 100 |

Other flags: `--train-split`, `--val-split` (test = 1 − train − val),
`--seed`, `--epochs`, `--lr-grid`, `--device`, `--top-k-eval`
(Hit-in-top-K metric width), `--loss {split_mse,rank}` (training
objective — see § *Differentiability and training*),
`--listnet-temperature` (ListNet target softmax temperature T in Å;
only used when `--loss rank`; default 5.0 — sweep
`{1.0, 3.0, 10.0}` to see how peakedness affects val Hit-in-top-K).

Output: `out/trained_params.pt` with the learned parameters, loss
history, the selected lr, the full grid results, and the exact
`train_proteins` / `val_proteins` / `test_proteins` split used.

### (2b) FFT docking search — `examples/04_fft_docking.py`

Pose **generation** in pure PyTorch, without the C/Fortran ZDOCK
binary. For each of N sampled rotations the FFT evaluates
`α S_SC + S_IFACE + β S_ELEC` at every translation cell in one batched
FFT pair, then returns the top-K (rotation, translation) poses. This
is the same function that `docking_score_elec` computes at a single
pose — the FFT path has been verified bit-exact against it (max
relative diff ~5 × 10⁻¹⁵ on float64 / 1KXQ).

```bash
# 64 random rotations, ~5 s on CUDA, ~3.5 min on CPU
uv run python examples/04_fft_docking.py

# Bigger rotation set on CUDA
CUDA_VISIBLE_DEVICES=0 uv run python examples/04_fft_docking.py \
    --device cuda --n-rotations 1024

# On Apple Silicon
uv run python examples/04_fft_docking.py --device mps --n-rotations 128
```

Output: `out/04_fft_docking_top.txt` — a table of the top-K
`(score, quat_idx, translation)` tuples. The script also re-scores
every reported pose via `docking_score_elec` and reports the maximum
relative diff; a pass threshold of ~10⁻¹⁰ on float64 / 10⁻⁴ on float32
is built in.

Low-level API — `zdock.search.docking_search(...)` takes the same
10-field receptor/ligand bundle as `docking_score_elec`, a rotation
grid `quaternions: (R, 4)`, and the learnable parameters `alpha /
iface_ij_flat / beta / charge_score_lut`. Returns a `DockingResultSC`
dataclass with `.scores (ntop,) .quat_indices (ntop,) .translations
(ntop, 3)`. Rotation grids can be produced via
`zdock.rotation_grid.random_quaternions(n, seed=…)` or
`euler_quaternions(deg=…)` — exact ZDOCK-table compatibility is
future work (see `PORT_PLAN_FFT.md`).

The search is end-to-end differentiable: `result.scores.sum().backward()`
propagates through the FFT to `alpha / iface / beta / charge_score_lut`
without NaN. Sparse gradient through `torch.topk`, so only poses in
the retained top-N contribute.

### (2c) Batched decoy generation over BM4 — `examples/05_fft_generate_decoys.py`

Iterates over every protein in `bm4_full.h5` and writes FFT-generated
decoys (top-N poses with scores, rotations, translations, and
reconstructed ligand coordinates) to an output h5. This is the
pure-PyTorch replacement for ZDOCK's decoy-generation step — no C /
Fortran binary required. Handles OOM by automatically halving
`rot_chunk_size`.

```bash
# Quick subset (3 proteins × 1024 rotations × top-500): ~15 s on A6000
uv run python examples/05_fft_generate_decoys.py --device cuda \
    --proteins 1PPE 2SIC 1R0R --n-rotations 1024 --ntop 500 \
    --out out/bm4_fft_decoys_smoke.h5

# Full BM4 with Julia defaults, random 4096-rotation sampling
uv run python examples/05_fft_generate_decoys.py --device cuda

# ZDOCK-comparable ~54k rotations (6° Euler), trained params
uv run python examples/05_fft_generate_decoys.py --device cuda \
    --euler-deg 6.0 --params-ckpt out/trained_params_rank.pt

# Parallelise across 4 GPUs — dynamic 1-complex-per-GPU pool.
# Proteins are dispatched to whichever GPU becomes free next, so
# no upfront chunk imbalance; each GPU processes one complex at a
# time with CUDA_VISIBLE_DEVICES isolation and per-worker tmp h5s
# that are merged into --out at the end.
uv run python examples/05_fft_generate_decoys.py \
    --gpus 0,1,2,3 --n-rotations 4096
```

Output layout (per protein group):
`lig_xyz_decoy (F, N_lig, 3)`,  `score (F,)`,  `rotation_quat (F, 4)`,
`translation (F, 3)`, and `rmsd_vs_bm4_best (F,)` if ground-truth
RMSD is available. Root attrs capture the rotation grid and scorer
params so downstream consumers know what produced the file.

### (2d) DockQ-driven self-consistent training — `06_train_dockq_fft.py`

A separate training path that lives **alongside** the canonical BM4
workflow (`02_train.py` / `03_evaluate.py` are preserved unchanged).
Motivation: the FFT search can find non-native poses that outscore
the near-native region, and those adversarial poses are absent from
the ZDOCK-curated BM4 decoy set — so training on BM4 alone leaves the
failure unfixed. This path uses **FFT-generated decoys as the
training distribution** plus **DockQ v2 as the ranking signal**.

**Step 1 — build FFT decoy dataset with DockQ labels**:

Two filter modes, chosen with `--filter-mode`:

- `top_k` (default): global top-N by score across every
  (rotation × translation) candidate. Fast, but biased toward what
  the current scorer considers best — with untrained parameters
  `n_positive` is often 0 (all decoys are non-native), and
  self-consistent training stalls because the rank loss has no
  positives to rank toward.
- `stratified`: mix of three sources that **guarantees positives
  plus broad score coverage**:
  1. **Near-native cone** (`--n-anchor`, default 200): for each of
     `n_anchor` rotations sampled within `--cone-deg` (default 12°)
     of the Kabsch-optimal rotation, place the ligand at the
     geometrically-matched translation. No FFT for this source —
     guarantees every anchor has DockQ ≥ 0.23 regardless of scorer
     state. Scores are recovered via `docking_score_elec`.
  2. **Hard negatives** (`--n-hard`, default 1000): top-K by score
     from uniform random rotations — the "FFT top-1 is non-native"
     failure mode captured explicitly in the training set.
  3. **Controls** (`--n-control`, default 200): tail of the same
     random source — lower-ranked but still explored poses, to
     prevent mode collapse around the hard-negative region.

```bash
# Recommended: stratified with near-native anchors
uv run python scripts/build_fft_decoys.py \
    --gpus 0,1,2,3 --filter-mode stratified \
    --n-anchor 200 --n-hard 1000 --n-control 200

# Subset for quick iteration
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_fft_decoys.py \
    --proteins 1PPE 2SIC 1R0R --filter-mode stratified \
    --out out/fft_decoys_smoke.h5

# Legacy top-K (= ZDOCK-style pure score ranking, no positive guarantee)
uv run python scripts/build_fft_decoys.py --gpus 0,1,2,3 \
    --n-rotations 4096 --ntop 2000    # filter-mode defaults to top_k
```

Typical stratified smoke on a trypsin–BPTI complex (1PPE):

```
anchors  (n=200)   score [-4.6e+02, -3.4e+02]   DockQ median 0.89   positives 200
hard-neg (n=1000)  score [+6.3e+02, +7.3e+02]   DockQ median 0.02   positives 0
controls (n=200)   score [+6.2e+02, +6.3e+02]   DockQ median 0.02   positives 0
```

Note how the untrained scorer rates native-like poses ~1000 units
LOWER than alternative sites — exactly the failure the rank + margin
losses are designed to correct.

Output schema per protein:
`lig_xyz (F, N_lig, 3)`, `score (F,)`, `rotation_quat (F, 4)`,
`translation (F, 3)`, plus **`fnat`**, **`i_rmsd`**, **`l_rmsd`**,
**`dockq`** — atom-level DockQ v2 (Mirabello & Wallner 2024) computed
against a pseudo-native reference (the BM4 decoy with smallest
stored RMSD, typically ~0.5–2 Å from crystal).

**Step 2 — train, comparing loss functions**:

```bash
# DockQ listwise ranking loss (softmax(dockq/T), T=0.2)
uv run python examples/06_train_dockq_fft.py \
    --decoys out/fft_decoys_dockq.h5 --loss dockq_rank

# Hard-negative margin loss: positives (DockQ ≥ 0.23 CAPRI-acceptable)
# must outscore every negative by at least the margin
uv run python examples/06_train_dockq_fft.py \
    --decoys out/fft_decoys_dockq.h5 --loss dockq_margin \
    --margin-positive-threshold 0.23 --margin 1.0

# For head-to-head comparisons, also run the existing losses on the
# same FFT-decoy data:
uv run python examples/06_train_dockq_fft.py --loss split_mse   # MSE baseline
uv run python examples/06_train_dockq_fft.py --loss rank        # ListNet on RMSD
```

The loss implementations live in `src/zdock/train.py`:

- `loss_listnet_dockq(scores, dockq, temperature)` — ListNet on DockQ.
- `loss_margin_hard_negatives(scores, dockq, positive_threshold, margin)`
  — hinge penalty whenever a DockQ-negative pose outscores `min(DockQ-positive scores)` by less than `margin`. Directly targets the "top scorer is non-native" failure mode.

Both are pure single-term losses so runs with `--loss dockq_rank` and
`--loss dockq_margin` are directly comparable. Combine them by
training sequentially (fine-tune margin on a rank-trained checkpoint)
or by extending the training loop externally.

**Step 3 — evaluate on the held-out test split**:

```bash
uv run python examples/07_evaluate_dockq.py \
    --params out/trained_params_dockq_fft.pt --split test
```

Reports top-1 DockQ before vs after, best-DockQ-in-top-K, CAPRI tier
transitions per protein, and success rates at Acceptable/Medium/High
thresholds. Contrast with `03_evaluate.py` (unchanged — reports
Hit-in-top-K on the BM4 decoy set).

### (3) Evaluate the held-out test split — `examples/03_evaluate.py`

Reads the ckpt from (2), pulls out `test_proteins` (and the `h5`,
`top_k` recorded at train time), reloads those proteins, and compares
**before** (default params) vs **after** (ckpt params) per-protein:

- Hit / Miss mean score
- Hit-in-top-K
- best (1-based) rank of any Hit pose
- ΔScore (Hit, Miss)

Plus a pooled before-vs-after scatter at `out/03_before_after.png`.

```bash
# Default: evaluate the ckpt at out/trained_params.pt on its test split
uv run python examples/03_evaluate.py

# Evaluate on the val split instead (useful when sanity-checking the lr pick):
uv run python examples/03_evaluate.py --split val

# Different ckpt / different HDF5 location:
uv run python examples/03_evaluate.py --params out/my_ckpt.pt --h5 datasets/bm4_full.h5
```

The script is read-only: running it many times on the same ckpt is
cheap and deterministic.

## Timing on 1KXQ, 10 poses, α=0.01, β=3.0

| Operation | CPU (float64, M2) | MPS (float32, M2) | Notes |
|---|---|---|---|
| Setup (SASA on 3908-atom receptor) | ≈ 2 s | ≈ 0.8 s | neighbor-packed |
| `docking_score_elec` forward / pose | ≈ 10 ms | ≈ 70 ms | MPS launch overhead dominates |
| `docking_score_elec` backward / pose | ≈ 18 ms | ≈ 100 ms | autograd |
| 200-epoch training (1KXQ, 10 poses) | ≈ 13 s | ≈ 40 s | loss drops ~78% |
| Full pytest suite | ≈ 18 s | ≈ 35 s | 58 tests |

MPS gains relative to CPU scale with F. At F=100 on a CUDA A100, expect
50–100× speedup over single-core CPU.

### Reproducing the CPU vs GPU comparison

`benchmark_gpu.py` sweeps F ∈ {10, 40, 100} on CPU and CUDA, reporting
forward and forward+backward timings plus the CPU/GPU speedup ratio. It
uses the phase5 1KXQ reference inputs and tiles the 10 ligand poses
along the frame axis to reach each target F.

```bash
# Pick an idle GPU (e.g. index 2) and run:
CUDA_VISIBLE_DEVICES=2 uv run python benchmark_gpu.py
```

Measured on an RTX A6000 (float32 on CUDA, float64 / OMP=1 on CPU):

| F   | CPU fwd | CPU fwd+bwd | CUDA fwd | CUDA fwd+bwd | speedup fwd | speedup fwd+bwd |
|----:|--------:|------------:|---------:|-------------:|------------:|----------------:|
|  10 |  361 ms |      326 ms |  16.4 ms |      20.3 ms |         22× |             16× |
|  40 |  976 ms |     1028 ms |  18.6 ms |      22.3 ms |         53× |             46× |
| 100 | 2482 ms |     2626 ms |  24.6 ms |      28.3 ms |        101× |             93× |

CUDA wall time is nearly flat in F (launch overhead dominates), so the
speedup grows with pose batch size.

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
  [FOLLOWUPS.md](FOLLOWUPS.md) — F-1 CHARMM19 charges, F-4
  `torch.compile`, …  (F-2 multi-protein training and F-3 test-set
  evaluation are now covered by `examples/02_train.py` /
  `examples/03_evaluate.py` on BM4.)
- **B4 physics note**: `../docking/tests/julia_ref/b4_physics_report.md`
  (Σq/Σr vs Σq/r) — superseded by the Phase C section of PORT_PLAN.md
- **Julia reference tests**: `../docking/tests/julia_ref/README.md`
- **Master's thesis**: `../docking/master_thesis/ICS-25M-23MM336.pdf`

## License

MIT — inherits from the parent repository.

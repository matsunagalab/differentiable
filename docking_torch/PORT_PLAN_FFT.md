# FFT docking search — audit notes and port plan

Companion to the parent repo's `PORT_PLAN.md` (B1–B15 scoring-path bugs).
This memo audits `../docking/docking.jl` lines 1530–1668 (the FFT-based
pose search) before porting to PyTorch. Trust posture: **Julia is a
reference with caveats, not a ground truth.** Authoritative sources in
decreasing order of trust are (1) Chen 2002 / 2003 / master's thesis,
(2) the existing PyTorch `docking_score_elec` (Phase-5 tested), (3) a
naive O(N³) direct cross-correlation written for tests, (4) Julia
`docking()`, (5) Julia `compute_docking_score_with_fft`.

## A1 — Two inconsistent FFT formulations (RESOLVED)

`docking()` uses
```julia
# line 1653
grid .= ifftshift(ifft(ifft(grid_sc_receptor) .* fft(grid_sc_ligand)))
score_sc .= (real(grid) .- imag(grid)) .* nxyz
```

`compute_docking_score_with_fft()` uses
```julia
# lines 1538/1543
t = ifft(fft(grid_RSC) .* conj.(fft(conj.(grid_LSC))))
score = real(t)
```

### Numerical verification (N=8 complex signals, Python ground-truth)

| Julia formula | Reduces to |
|---|---|
| `docking()` (f1): `ifft(ifft(R) * fft(L)) * N` | `Σ_m R[m] · L[m+t]` |
| `compute_docking_…` (f2): `ifft(fft(R) · conj(fft(conj(L))))` | `Σ_m R[m] · L[m−t]` |
| Standard conj-xcorr: `ifft(fft(R) · conj(fft(L)))` | `Σ_m R[m] · conj(L[m−t])` |

f1 and f2 are mirror images along the translation axis; neither equals the
standard `conj`-xcorr for **complex** inputs. For **real** inputs,
standard conj-xcorr equals f2.

### Status

- **f2 is dead code.** Grep of the full Julia repo shows `compute_docking_score_with_fft` is never called (only referenced in a stale comment in `tests/julia_ref/generate_refs.jl`). Ignore.
- **f1 is consistent** with its decoder (`generate_ligand` subtracting `dx = (peak_idx − center) · spacing` from ligand coords). Peak at `+t0` means "ligand in its current pose is offset +t0 from the receptor binding site," and the decoder applies `−t0` to snap it back. The sign chain closes correctly.

### Resolution for the PyTorch port

**Do not mirror Julia's f1 + compensating-negate idiom.** Instead, use the
textbook conj-xcorr convention:

```python
score_grid = torch.fft.ifftn(torch.fft.fftn(R) * torch.fft.fftn(L).conj())
# for real R, L: score_grid[t] = Σ_m R[m] · L[m - t] = "ligand shifted by +t"
# decoder: ligand_xyz += peak_offset (ADD, not subtract)
```

This is the standard `scipy.signal.correlate` / `torch.nn.functional.conv3d`
convention and is less error-prone than Julia's approach. It also operates
on **real-valued** grids for each term separately (see A3 resolution).

## A2 — `ifftshift` vs `fftshift` on output (deferred to Phase 1 test)

Julia wraps the output in `ifftshift` and the decoder uses
`ix_center = ceil(nx/2) + 1` (1-indexed) for unshifting. `fftshift` and
`ifftshift` differ only for odd N (they are identical for even N), so the
correct shift for PyTorch depends on the grid size parity.

**Resolution for Phase 1**: unit test — place a single ligand atom at a
known Cartesian offset (dx, dy, dz) from a single receptor atom, run the
FFT search at `ntop=1`, verify that the decoded translation recovers
(dx, dy, dz) to within `spacing` tolerance. Test both even and odd grid
sizes. Off-by-one bugs will surface immediately.

## A3 — SC score combination `(real − imag) · nxyz` (DEVIATE)

Julia combines the complex cross-correlation output as
```julia
score_sc = (real(grid) − imag(grid)) · nxyz
```
with `grid_sc = grid_real + i·grid_imag`, where `grid_real` is the surface
envelope and `grid_imag` is the core-penalty weight (3.5 surface / 12.25
core per Chen 2002).

### Algebra

Let `R = R_r + i R_i`, `L = L_r + i L_i` (all real grids). Using f1, the
complex xcorr `c[t] = Σ_m R[m] · L[m+t]` gives:

- `real(c[t]) = xcorr(R_r, L_r)[t] − xcorr(R_i, L_i)[t]`
- `imag(c[t]) = xcorr(R_r, L_i)[t] + xcorr(R_i, L_r)[t]`
- `real − imag = xcorr(R_r, L_r) − xcorr(R_i, L_i) − xcorr(R_r, L_i) − xcorr(R_i, L_r)`
- `= 2·xcorr(R_r, L_r) − xcorr(R_r + R_i, L_r + L_i)`

This is `2·(surface-surface overlap) − (total-shape overlap)`. It is a
valid SC score only if `R_r` and `R_i` are specifically encoded so that
this combination matches Chen 2002 — which is fragile to verify and
surprising to a reader.

### Resolution for the PyTorch port

**Decompose explicitly.** Build four real grids and combine per the
physics in `score.py::_assign_sc_*` (already Phase-5 tested against the
Julia reference for the re-scoring path):

```
R_surface, R_core      (real grids from receptor atoms by SASA class)
L_surface, L_core      (real grids from ligand atoms by SASA class)

sc[t] = α_SS · xcorr(R_surface, L_surface)[t]
       − β_CC · xcorr(R_core,    L_core   )[t]
       + (any cross terms per Chen 2002)
```

Exact coefficients must be derived to match `docking_score_elec`'s SC
term; see **Verification test V-SC** below. Avoiding Julia's complex-grid
trick makes each term's physical meaning visible, avoids a whole class of
sign bugs, and lets us verify termwise against the already-tested
PyTorch re-scoring path.

## A4 — DS coefficient `0.5 · imag(...) · nxyz` (DEFERRED)

`docking.jl:1659`:
```julia
score_ds = 0.5 · imag(grid_ds_ligand) · nxyz
```

The `0.5` is unexplained. Three candidates:

1. **Physics**: 0.5 comes from a double-counting correction in the Chen
   desolvation formulation (pair energies counted once, not twice).
2. **Historical**: tweaked empirically to balance SC and DS magnitudes
   during an earlier tuning pass.
3. **Bug**: a stray scaling nobody audited.

**Resolution**: port the DS term WITHOUT the `0.5` initially, then compare
the FFT-computed DS score against `docking_score_elec` on a known pose.
If they match without the factor, Julia's 0.5 was a bug or empirical
tweak — document, proceed without it. If they differ by exactly 2×, the
0.5 is needed — adopt with a comment citing Chen 2003 eq. X. **Do not
transcribe magic numbers from Julia without a test.**

(Note: our plan folds DS into the FFT search, whereas Julia's f1 is the
only place DS appears in the search path. The re-scoring `docking_score_elec`
in the PyTorch port does not currently include a separate DS term — ELEC
is the nearest analogue. Confirm whether "DS" and "ELEC" are the same
quantity under different names, or genuinely distinct, before porting.)

## A5 — `compute_docking_score_with_fft` dead-code check (RESOLVED)

`grep -rn compute_docking_score_with_fft .jl .ipynb` returns only the
definition and one comment line in `tests/julia_ref/generate_refs.jl`.
Never called. Ignore for the port.

## A6 — Interaction with known PORT_PLAN bugs (RESOLVED)

- **B10 (ELEC not Coulomb), B11 (charge sign), B13 (V=0 inside core)**:
  The Julia FFT `docking()` doesn't include ELEC, so those bugs don't
  appear in the search path. Our plan DOES fold ELEC into the FFT; we
  use the already-fixed Coulomb-mode V grid from
  `score.py::_assign_*_elec` (Phase-C tested), **not** anything from
  the Julia FFT.
- **B2 (loss drops five of six terms)**: search-path-unrelated, but a
  reminder that superficially reasonable Julia code can be wrong.

## A9 — Latent bug in existing `_assign_sc_minus(receptor=False)` (DISCOVERED during Phase 1 port)

While writing `docking_search_sc`, I initially reused
`src/zdock/score.py::_assign_sc_minus(receptor=False)` to build the
ligand's imaginary SC grid. V-SC produced ~2× off scores.

Investigation: `_assign_sc_minus(receptor=False)` writes weight **12.25**
to every cell near a surface atom. But `docking_score_elec` builds the
ligand imag grid inline (score.py lines 299-313) using **3.5** for
surface atoms and **12.25** only for core atoms. The values disagree.

Status: latent bug. `_assign_sc_minus(receptor=False)` is **dead code**
(`grep` shows only `receptor=True` is invoked anywhere in the repo), so
no currently-shipping numbers are affected. But the function is public-
ish (underscore-prefixed, so internal, but importable) and any future
caller using `receptor=False` would silently get wrong SC values.

Resolution in this port: `search.py` provides
`_build_ligand_sc_grid_single` which mirrors `docking_score_elec`'s
inline construction exactly. Verified bit-exact agreement across all
phase-1 cross-checks (V-SC, V-ROT, V-ODD, V-1KXQ).

Follow-up: either delete `_assign_sc_minus`'s `receptor=False` branch,
or fix it to write 3.5/12.25 matching `docking_score_elec`. Tracked as
out-of-scope for this port (no shipping code touches it).

## A8 — Cyclic-wraparound when ligand is translated beyond the grid padding (DISCOVERED during V-ODD cross-check)

The FFT produces **cyclic** cross-correlation. For a translation `t`
that pushes ligand atoms outside the grid bounds, the cyclic output
"wraps" those atoms back to the opposite face of the grid, producing a
spurious overlap with the receptor that does not exist physically.

`docking_score_elec` handles the same situation differently: its scatter
routines bounds-check each atom and discard out-of-grid contributions,
so the physical score at such translations is 0.

### Demonstration

Seed-11 synthetic case, grid 4×5×4, translation `t=(+2,-1,+1) cells` =
`(+6,-3,+3) Å`. Ligand extent after translation is `[5.4, 6.5]` in x,
but grid x range is `[-5.58, +3.42]`:

| Method | Score at t=(2,-1,1) |
|---|---|
| `docking_score_elec` (physical) | `0.0` |
| `docking_search_sc` (cyclic FFT) | `-1.8e+01` (aliased) |

### Resolution

For Phase 1, this is documented as a known limitation. In-bounds
translations (where every ligand atom stays inside the grid) match
`docking_score_elec` to float-precision (V-ODD v2 pass across 8 seeds ×
27 translations each, max diff ~1e-13). For large proteins the default
`generate_grid` padding (ligand x-extent) is typically enough to cover
all physically meaningful translations; the issue only bites small
synthetic cases and edge-of-grid poses.

A cleaner fix for Phase 3 (out of Phase 1 scope): use a **search-
specific grid that is zero-padded to `N + lig_extent_cells` on each
axis** before FFT. This converts circular cross-correlation into
linear cross-correlation, eliminating wraparound. Cost: grid gets
~2× larger, so memory / FFT time roughly 2–4× up.

The existing `generate_grid` already pads by ligand x-extent but not by
the ligand's full 3-D extent on all axes — the asymmetry of A7 means
y- and z-axis padding comes from the x-extent, which is
under-specified for non-spherical ligands. Fixing A7 (isotropic
padding) would partially mitigate A8 at small extra grid cost.

## A7 — Grid asymmetry (KNOWN; keep for now)

`generate_grid` pads each axis by the ligand's **x-axis extent** only, yielding
an anisotropic box biased by the receptor shape. Existing
`geom.py::generate_grid` preserves this for parity. For the FFT search,
this is mildly wasteful but not incorrect. **Keep** for Phase 1; consider
switching to isotropic padding in a follow-up if it simplifies the code
or shrinks memory.

## Execution phasing

1. **Phase 1 — SC-only port**: implement `docking_search` in
   `src/zdock/search.py` using standard conj-xcorr on four real grids
   (A1 + A3 resolutions). Phase-1 verification:
   - **V-direct**: FFT result matches naive O(N³) direct cross-correlation
     (tolerance 1e-6 on float64) on a 16³ synthetic case. Catches FFT
     convention and ifftshift bugs (A2).
   - **V-SC**: FFT-computed SC at a known 1KXQ pose matches
     `docking_score_elec(single_pose)` SC term to 1e-6 (float64). Locks
     in the sign / scaling of the SC combination.
2. **Phase 2 — add DS, IFACE, ELEC** (one at a time, each with its own
   single-pose match test against `docking_score_elec`). A4 gets resolved
   by the DS match test.
3. **Phase 3 — performance and demo** on 1KXQ at deg=6; compare against
   Julia `docking()` and upstream ZDOCK `.out`. Discrepancies flag further
   Aₙ items, not automatic changes.

Each phase is its own commit so `git bisect` is cheap.

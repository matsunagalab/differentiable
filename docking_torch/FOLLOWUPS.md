# Optional follow-up tasks

The core port (PyTorch + Julia + literature-correct Coulombic ELEC +
28 green tests on CPU / MPS) is complete as of commit `821f553`. Nothing
below is required to run / train / evaluate the model; each item raises
either physical fidelity or runtime performance by a further notch.

See `../docking/PORT_PLAN.md` for the completed bug list (B1 – B15) and
the design rationale for why the ELEC path was rewritten.

---

## F-1: Full CHARMM19 per-atom partial-charge table (B14 proper fix)

**Summary.** Replace the 11-entry `charge_score` LUT with a CHARMM19
partial-charge table keyed by `(resname, atomname)` so every heavy
atom carries its paper-faithful charge instead of collapsing to one
of 11 bucket values.

**Current state.** `zdock.atomtypes.partial_charge_per_atom(charge_ids,
charge_score_lut)` expands the 11-entry LUT. Atoms that don't match a
side-chain rule in `set_charge` fall into bucket 8 (`CA`, `q=0`), so
most backbone / aromatic carbons contribute zero to the Coulomb sum.
This was flagged as bug B14 in `docking/PORT_PLAN.md`; we kept the
lightweight LUT as a stopgap.

**Work items.**
1. Obtain a CHARMM19 parameter set (e.g. `toph19.inp` / `par19.inp`
   from the CHARMM distribution).
2. Parse `RESI … ATOM name TYPE charge …` records into a Python dict
   `{(resname, atomname): charge}`.
3. Add `zdock.atomtypes.set_partial_charge(resnames, atomnames,
   device, dtype) -> Tensor` (parallels `set_atomtype_id`).
4. Thread it through `docking_score_elec` so a per-atom `q` tensor
   overrides the `charge_score`-bucket path when supplied.
5. Tests: one new `test_physics.py` assertion that the summed receptor
   charge for 1KXQ is close to the expected net protein charge at
   neutral pH.

**Dependencies.** None.

**Estimate.** ~1 day (LUT generation + plumbing + test).

**Verification.** 1KXQ receptor net charge (Σ per-atom charges) should
land in the expected integer band for that protein at pH 7; score_elec
order of magnitude comparable to published ZDOCK outputs on the CAPRI
benchmark.

---

## F-2: Full three-protein training (reproduce thesis Fig 5.1–5.5)

**Summary.** Train α, β, iface, charge_score on 1KXQ + 1F51 + 2VDB
simultaneously for 200 epochs and compare the loss curve / final
parameters against the thesis numbers.

**Current state.** `tests/test_phase7_train.py::test_train_200_epoch_1kxq`
trains on 1KXQ only and reports 78 % loss reduction. 1F51 and 2VDB
decoy trajectories exist in `../docking/protein/{1F51,2VDB}/`, but
they are not wired into the Julia reference generator.

**Work items.**
1. Extend `docking/tests/julia_ref/generate_refs.jl` so it emits
   `phase5_1F51.h5` and `phase5_2VDB.h5` with the same schema (prepared
   rec/lig inputs + coulomb / legacy scores).
2. Add `scripts/train_full.py` (or a `@pytest.mark.slow` test) that
   loads all three proteins into a list of `ProteinInputs` and runs
   `zdock.train.train(proteins=[p1, p2, p3], n_epoch=200)`.
3. Log per-epoch loss + the 157 parameter values; compare to thesis
   Figure 5.1 (loss curve 4.32×10⁴ → ~1.52×10⁴) and Figure 5.5
   (Hit / Rank table).

**Dependencies.** None.

**Estimate.** 0.5 day for plumbing + 3–5 hours CPU runtime for the
200-epoch fit (less on MPS / CUDA).

**Verification.** Loss curve qualitatively matches Fig 5.1. α drifts
toward the thesis sign flip (0.01 → negative). β moves in the expected
direction — Coulomb mode may give a different quantitative endpoint
than the thesis legacy mode since the ELEC term is now physical.

---

## F-3: Test-protein evaluation (reproduce thesis Fig 5.6–5.8)

**Summary.** Score decoys of proteins **not** seen during training
(1CGI, 1ZHI) with the trained parameters and measure Rank / Hit
improvements.

**Current state.** No evaluation harness for held-out proteins.
`../docking/protein/{1CGI,1ZHI}/` contain decoy sets but no script
consumes them.

**Work items.**
1. Add `scripts/evaluate_test_proteins.py` that:
   - Loads a trained-parameter checkpoint (output of F-2).
   - Runs `docking_score_elec` on 1CGI / 1ZHI top-N decoys.
   - Computes Hit = count(top-100 with RMSD ≤ 2.5 Å) and Rank =
     first-hit index after sorting by score descending.
2. Repeat for `elec_mode="legacy"` and `elec_mode="coulomb"` to see
   whether the physical Coulomb path generalizes better.
3. Compare to thesis Fig 5.8 (1CGI Rank 86 → 77, 1ZHI 130 → 111).

**Dependencies.** F-2 (needs trained parameters).

**Estimate.** 1 day (score + sort + Rank / Hit computation + comparison
report).

**Verification.** Ranks drop (improve) after training vs. pre-training
baseline. Ideally the coulomb mode's generalization ≥ legacy mode's.

---

## F-4: `torch.compile` for MPS / CUDA speedup

**Summary.** Wrap `docking_score_elec` in `torch.compile` to fuse
scatter + reduction kernels and reduce Python-side launch overhead on
accelerators.

**Current state.** Forward time on 1KXQ × 10 poses: CPU float64 ~9.2
ms/pose, MPS float32 ~45 ms/pose. MPS is slower than CPU because each
frame triggers several small kernels whose launch overhead dominates.

**Work items.**
1. Decorate `docking_score_elec` (or just the batched ELEC / IFACE
   loops) with `@torch.compile(dynamic=True, mode="max-autotune")`.
2. Identify graph breaks using `TORCH_LOGS=+dynamo uv run pytest
   tests/test_phase5.py`; likely culprits are `.item()` calls in
   `_neighbors_indices` and the `ceil` + clamp in cell-index math.
3. Move any remaining `.item()` accesses out of the compiled region
   by passing grid metadata (spacing, origin, shape) as plain Python
   floats into a refactored entry point.
4. Re-run `uv run pytest -q` to confirm 28/28 still green and ELEC
   stays bit-exact against Julia (test_phase5 / test_phase6).
5. Benchmark before / after on MPS and on a CUDA host (A100 / L40S).

**Dependencies.** None.

**Estimate.** 1 day (bulk of the time is chasing graph breaks).

**Verification.** Same inputs give scores within the existing tolerance
thresholds. MPS per-pose latency drops below CPU (expected target: ≤
10 ms / pose). CUDA should see 5–10× over CPU at F=100.

---

## Not doing (explicitly out of scope)

- **Rewriting Julia `rrule`** for `docking_score_elec_coulomb`. The
  PyTorch autograd path makes a hand-written rrule redundant; the old
  rrule had its own bugs (B5 / B6 / B7) and is only kept for legacy
  reproducibility.
- **GPU support on the Julia side.** `docking.jl` has a partial CUDA
  path; we rely on the PyTorch port for all accelerator work.
- **Full CAPRI benchmark integration** (Chen 2003 Table I). Interesting
  for a separate study; requires downloading the full 48 test-case
  benchmark and considerable wall time.

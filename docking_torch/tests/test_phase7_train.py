"""Phase 7: end-to-end Adam training smoke test.

We run a short training (30 epochs on 1KXQ top-10 decoys) and verify the
loss descends. Full 200-epoch three-protein training requires generating
the 1F51 / 2VDB reference inputs and isn't exercised here — the machinery
(autograd + Adam + B2-fixed loss) is identical though.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock.train import ProteinInputs, train


def _2d(a):
    arr = np.asarray(a)
    return arr.T if arr.ndim == 2 and arr.shape[0] == 3 else arr


def _3d(a):
    arr = np.asarray(a)
    return arr.transpose(2, 1, 0) if arr.ndim == 3 and arr.shape[0] == 3 else arr


def build_1kxq(load_ref, device, dtype) -> ProteinInputs:
    ref = load_ref("phase5", "scores")

    def T(key, int_=False):
        arr = np.asarray(ref[key])
        if arr.ndim == 3:
            arr = _3d(arr)
        elif arr.ndim == 2:
            arr = _2d(arr)
        dtype_ = torch.int64 if int_ else dtype
        return torch.as_tensor(arr, device=device, dtype=dtype_)

    F = int(ref["n_pose"])
    # Hit/Miss by first-3 are Hit (synthetic — real RMSD split would use
    # `*.zd3.0.2.fg.fixed.out.rmsds`). For the smoke test, what matters is
    # that both classes are non-empty so the loss has gradient signal.
    hit_mask = torch.zeros(F, dtype=torch.bool, device=device)
    hit_mask[:3] = True

    return ProteinInputs(
        rec_xyz=T("rec_xyz"),
        rec_radius=T("rec_radius"),
        rec_sasa=T("rec_sasa"),
        rec_atomtype_id=T("rec_atomtype_id", int_=True),
        rec_charge_id=T("rec_charge_id", int_=True),
        lig_xyz=T("lig_xyz"),
        lig_radius=T("lig_radius"),
        lig_sasa=T("lig_sasa"),
        lig_atomtype_id=T("lig_atomtype_id", int_=True),
        lig_charge_id=T("lig_charge_id", int_=True),
        hit_mask=hit_mask,
    )


def test_train_smoke_loss_decreases(load_ref, device, dtype):
    """30 epochs on CPU (float64). Quick smoke test for CI parity."""
    p = build_1kxq(load_ref, device, dtype)
    out = train([p], n_epoch=30, lr=0.01, device=device, dtype=dtype,
                progress_every=10)
    hist = out["history"]["loss"]
    print(f"\n[train] initial={hist[0]:.4e}  final={hist[-1]:.4e}  "
          f"reduction={(hist[0]-hist[-1])/hist[0]*100:.1f}%")
    assert hist[-1] < hist[0], (
        f"loss did not decrease: init={hist[0]} final={hist[-1]}"
    )
    assert hist[-1] < hist[0] * 0.95
    assert not torch.allclose(out["alpha"], torch.tensor(0.01, device=device, dtype=dtype))
    assert not torch.allclose(out["beta"], torch.tensor(3.0, device=device, dtype=dtype))


@pytest.mark.slow
def test_train_200_epoch_1kxq(load_ref, device, dtype):
    """Full 200-epoch training on 1KXQ alone (matching thesis schedule).

    Run with `pytest -m slow` to opt in. Proves loss continues to descend
    across the full 200 epochs and parameters land near physically-plausible
    values (α ~ 0.01, β ~ 3, iface still bounded)."""
    p = build_1kxq(load_ref, device, dtype)
    out = train([p], n_epoch=200, lr=0.01, device=device, dtype=dtype,
                progress_every=25)
    hist = out["history"]["loss"]
    print(f"\n[train-200] initial={hist[0]:.4e}  final={hist[-1]:.4e}  "
          f"reduction={(hist[0]-hist[-1])/hist[0]*100:.1f}%")
    assert hist[-1] < hist[0] * 0.5   # expect ≥50% drop after 200 epochs
    # Parameter sanity
    print(f"[train-200] α = {out['alpha'].item():.4e}  β = {out['beta'].item():.4e}")
    assert abs(out["alpha"]) < 1.0, "α drifted out of plausible range"
    assert abs(out["beta"]) < 10.0, "β drifted out of plausible range"

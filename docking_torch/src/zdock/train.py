"""Adam training loop for the 157 scoring parameters (α, β, iface_flat,
charge_score), with B2 loss-function bug fixed.

The Julia notebook (cell 62) wrote the loss as:

    l = sum(...) / 9
      + sum(...) / 91
      + ...

which — due to Julia's expression-continuation rules — only assigned the
first term to `l`; the `+` lines started fresh expressions and were
discarded. The port below puts the six MSE terms inside a single `sum()`
call so all of them contribute to the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .atomtypes import charge_score as default_charge_score
from .atomtypes import iface_ij
from .score import docking_score_elec


@dataclass
class ProteinInputs:
    """Bundle of prepared inputs for one receptor–ligand pair, as
    docking_score_elec expects them (see `score.py` docstring for the
    preconditions on decenter / mass init / charge id)."""
    rec_xyz: torch.Tensor
    rec_radius: torch.Tensor
    rec_sasa: torch.Tensor
    rec_atomtype_id: torch.Tensor
    rec_charge_id: torch.Tensor
    lig_xyz: torch.Tensor              # (F, N_lig, 3)
    lig_radius: torch.Tensor
    lig_sasa: torch.Tensor
    lig_atomtype_id: torch.Tensor
    lig_charge_id: torch.Tensor
    hit_mask: torch.Tensor             # (F,) bool: True for Positive poses

    def call(
        self,
        alpha: torch.Tensor,
        iface: torch.Tensor,
        beta: torch.Tensor,
        charge: torch.Tensor,
    ) -> torch.Tensor:
        return docking_score_elec(
            self.rec_xyz, self.rec_radius, self.rec_sasa,
            self.rec_atomtype_id, self.rec_charge_id,
            self.lig_xyz, self.lig_radius, self.lig_sasa,
            self.lig_atomtype_id, self.lig_charge_id,
            alpha, iface, beta, charge,
        )


def make_ideal_targets(initial_scores: torch.Tensor, hit_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Julia's "ideal score distribution" scheme: all Hit (Positive) poses
    get the *maximum* of the pre-training score distribution as their
    target (we want them ranked highest), all Miss (Negative) poses get
    the *minimum* target. Returns (hit_target, miss_target) per pose —
    scalars broadcast to match."""
    hit_target = initial_scores.max()
    miss_target = initial_scores.min()
    return hit_target, miss_target


def loss_b2_fixed(
    scores: torch.Tensor,
    hit_mask: torch.Tensor,
    hit_target: torch.Tensor,
    miss_target: torch.Tensor,
) -> torch.Tensor:
    """Two-term MSE split by Hit/Miss with per-class normalization. This
    is the per-protein loss; the full Julia loss sums three such
    per-protein contributions (six terms total).

    B2 fix: make sure *both* sides actually contribute to the returned
    tensor. In the original Julia notebook the `+` at the start of each
    line broke expression continuation, so only the first term was kept.
    """
    hit_sq = ((scores[hit_mask] - hit_target) ** 2)
    miss_sq = ((scores[~hit_mask] - miss_target) ** 2)
    # Use mean per class so protein size doesn't bias the gradient;
    # sum is also valid — the Julia notebook used `/ n_hit` and `/ n_miss`
    # constants, equivalent to `mean()` here.
    hit_term = hit_sq.mean() if hit_sq.numel() > 0 else torch.zeros((), device=scores.device, dtype=scores.dtype)
    miss_term = miss_sq.mean() if miss_sq.numel() > 0 else torch.zeros((), device=scores.device, dtype=scores.dtype)
    return hit_term + miss_term


def total_loss(
    proteins: Iterable[ProteinInputs],
    alpha: torch.Tensor,
    iface: torch.Tensor,
    beta: torch.Tensor,
    charge: torch.Tensor,
    targets: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Sum per-protein losses. `targets[i]` = (hit_target, miss_target)
    for proteins[i]."""
    parts = []
    for p, (ht, mt) in zip(proteins, targets):
        scores = p.call(alpha, iface, beta, charge)
        parts.append(loss_b2_fixed(scores, p.hit_mask, ht, mt))
    return torch.stack(parts).sum()


def train(
    proteins: list[ProteinInputs],
    *,
    n_epoch: int = 200,
    lr: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    progress_every: int = 10,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the full Adam optimization and return the trained parameters
    plus a loss history. Targets are frozen at their pre-training values
    (the "ideal score distribution" scheme)."""
    # Initial parameters — same as Julia `train_param-apart.ipynb` cell
    # 27-28 defaults.
    alpha = torch.tensor(0.01, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
    iface_init = iface_ij(device=device, dtype=dtype, flat=True).clone()
    iface = iface_init.detach().requires_grad_(True)
    charge_init = default_charge_score(device=device, dtype=dtype).clone()
    charge = charge_init.detach().requires_grad_(True)

    # Freeze "ideal" targets from pre-training scores.
    with torch.no_grad():
        targets = []
        for p in proteins:
            s0 = p.call(alpha, iface, beta, charge)
            targets.append(make_ideal_targets(s0, p.hit_mask))

    opt = torch.optim.Adam([alpha, beta, iface, charge], lr=lr)
    history = {"loss": []}

    for epoch in range(n_epoch):
        opt.zero_grad()
        l = total_loss(proteins, alpha, iface, beta, charge, targets)
        l.backward()
        opt.step()
        history["loss"].append(float(l.detach()))
        if epoch % progress_every == 0 or epoch == n_epoch - 1:
            log(f"epoch {epoch:4d}  loss={float(l.detach()):.6e}")

    return {
        "alpha": alpha.detach(),
        "beta": beta.detach(),
        "iface": iface.detach(),
        "charge": charge.detach(),
        "history": history,
    }

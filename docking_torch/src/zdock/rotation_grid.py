"""Quaternion grid generators for the FFT docking search.

ZDOCK 3.0.2 uses a specific Euler-angle table at 6° spacing. This
module provides simpler drop-in alternatives that are adequate for
demonstration and experimentation:

  * `random_quaternions` — N uniformly-distributed unit quaternions
    (Gaussian-sampled then normalised, which is uniform on SO(3)).
  * `euler_quaternions` — (φ, θ, ψ) ZYZ Euler grid at `deg` spacing.

Exact ZDOCK-table reproduction is a future refinement (see
PORT_PLAN_FFT.md); the `rotate` convention matches `geom.rotate`
(docking.jl lines 1469–1490) so poses are compatible with the rest
of this package.
"""

from __future__ import annotations

import math

import torch


def random_quaternions(
    n: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return `(n, 4)` uniformly-distributed unit quaternions.

    Uses Gaussian 4-vectors normalised to unit norm — a standard way
    to sample SO(3) uniformly (Shoemake 1992).
    """
    # Always sample on CPU (torch CUDA RNG is streaming-only; a seeded
    # CPU generator gives reproducible results across backends), then
    # move to the target device.
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    if device is not None:
        q = q.to(device)
    return q


def euler_quaternions(
    deg: float = 15.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """ZYZ Euler-angle grid at `deg` spacing, converted to
    `geom.rotate`-compatible quaternions.

    Enumerates (φ, θ, ψ) with φ ∈ [0, 360), θ ∈ [0, 180], ψ ∈ [0, 360).
    Does NOT de-duplicate orientations that coincide at θ = 0/180
    (gimbal-lock); callers that care can unique-ify by the resulting
    rotation matrix. Coverage is denser near the poles — a known
    Euler limitation; `random_quaternions` avoids this.

    Returns `(R, 4)` unit quaternions in the convention used by
    `geom.rotate` (q1..q4 → Julia docking.jl rotate!).
    """
    if deg <= 0.0 or deg > 180.0:
        raise ValueError(f"deg must be in (0, 180], got {deg}")
    rad = math.radians(deg)
    # Number of samples per axis.
    n_phi = int(round(360.0 / deg))
    n_theta = int(round(180.0 / deg)) + 1     # endpoints included
    n_psi = int(round(360.0 / deg))
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=dtype)[:-1]
    theta = torch.linspace(0.0, math.pi, n_theta, dtype=dtype)
    psi = torch.linspace(0.0, 2.0 * math.pi, n_psi + 1, dtype=dtype)[:-1]

    # Cartesian product, shape (R, 3)
    ph, th, ps = torch.meshgrid(phi, theta, psi, indexing="ij")
    ph, th, ps = ph.reshape(-1), th.reshape(-1), ps.reshape(-1)

    # Convert ZYZ Euler to quaternion. Use the convention that matches
    # `geom.rotate`'s q1..q4 decomposition.
    #
    # We assemble q = q_z(ψ) · q_y(θ) · q_z(φ) where q_axis(a) is the
    # half-angle rotation around that axis, then map to (q1, q2, q3, q4)
    # = (x, y, z, w). This is standard Hamilton composition.
    def q_z(a):
        z = torch.zeros_like(a)
        return torch.stack([z, z, torch.sin(a / 2), torch.cos(a / 2)], dim=-1)

    def q_y(a):
        z = torch.zeros_like(a)
        return torch.stack([z, torch.sin(a / 2), z, torch.cos(a / 2)], dim=-1)

    def quat_mul(a, b):
        # Hamilton product, both in (x, y, z, w) convention.
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        return torch.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dim=-1)

    q = quat_mul(q_z(ps), quat_mul(q_y(th), q_z(ph)))
    q = q / q.norm(dim=-1, keepdim=True)
    if device is not None:
        q = q.to(device)
    return q

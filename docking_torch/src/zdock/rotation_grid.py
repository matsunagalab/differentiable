"""Quaternion grid generators for the FFT docking search.

ZDOCK 3.0.2 uses a specific Euler-angle table at 6° spacing. This
module provides simpler drop-in alternatives that are adequate for
demonstration and experimentation:

  * `random_quaternions` — N uniformly-distributed SO(3) rotations
    via `scipy.spatial.transform.Rotation.random`.
  * `euler_quaternions` — (φ, θ, ψ) ZYZ Euler grid at `deg` spacing.

Exact ZDOCK-table reproduction is a future refinement (see
PORT_PLAN_FFT.md).

**Quaternion convention**: this module returns quaternions in the
format consumed by `geom.rotate` (= `docking.jl::rotate!`), which is
the *inverse* of scipy's active-rotation convention. For a uniform
sampler this is immaterial — the inverted distribution is still
uniform on SO(3). If you need to interoperate with scipy's
`Rotation.apply` outcome-by-outcome, apply the quaternion conjugate
`(x, y, z, w) → (−x, −y, −z, w)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation


def random_quaternions(
    n: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return `(n, 4)` uniformly-distributed SO(3) quaternions via
    `scipy.spatial.transform.Rotation.random`.

    Output is in `geom.rotate` / Julia `rotate!` convention (see
    module docstring). For uniform sampling the convention choice
    has no effect on the distribution.
    """
    r = _ScipyRotation.random(n, random_state=seed)
    q = r.as_quat()  # (n, 4) scalar-last (x, y, z, w)
    q = torch.as_tensor(q, dtype=dtype)
    if device is not None:
        q = q.to(device)
    return q


def scipy_rotations_to_quaternions(
    r: _ScipyRotation,
    *,
    as_inverse: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a `scipy.spatial.transform.Rotation` (single or batch)
    to `(N, 4)` quaternions for use with `geom.rotate`.

    `as_inverse=True` (default) emits the quaternion conjugate of
    scipy's output so that `geom.rotate(v, q)` produces the same
    rotated vector as `r.apply(v)`. Set to False to pass the raw
    scipy quaternions (equivalent to applying the inverse rotation
    under our convention — fine for uniform samplers but not for
    known oriented rotations).
    """
    q = r.as_quat().reshape(-1, 4)
    if as_inverse:
        q = q.copy()
        q[:, :3] *= -1  # quaternion conjugate = inverse of unit quat
    q_t = torch.as_tensor(np.ascontiguousarray(q), dtype=dtype)
    if device is not None:
        q_t = q_t.to(device)
    return q_t


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

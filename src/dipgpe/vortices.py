"""Vortex detection: points in 2D fields, line piercings in 3D fields."""

from __future__ import annotations

import math

import torch


def _winding(psi, dims=(-2, -1)):
    """Phase winding (integer) of each plaquette (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1)
    in the plane of ``dims`` = (first, second) axis.

    The loop is counterclockwise in (first, second), so psi ~ first + i second has
    winding +1.
    """
    a = psi
    b = torch.roll(psi, -1, dims=dims[0])
    c = torch.roll(b, -1, dims=dims[1])
    d = torch.roll(psi, -1, dims=dims[1])
    total = (torch.angle(b * a.conj()) + torch.angle(c * b.conj())
             + torch.angle(d * c.conj()) + torch.angle(a * d.conj()))
    return torch.round(total / (2 * math.pi)).to(torch.int64)


def _bilinear_zero(a, b, c, d, iterations=8):
    """Zero (u, v) in [0, 1]^2 of the bilinear interpolant through the plaquette corners."""
    u = torch.full(a.shape, 0.5, dtype=torch.float64, device=a.device)
    v = torch.full_like(u, 0.5)
    a, b, c, d = (z.to(torch.complex128) for z in (a, b, c, d))
    for _ in range(iterations):
        f = a * (1 - u) * (1 - v) + b * u * (1 - v) + c * u * v + d * (1 - u) * v
        fu = (b - a) * (1 - v) + (c - d) * v
        fv = (d - a) * (1 - u) + (c - b) * u
        det = fu.real * fv.imag - fv.real * fu.imag
        det = torch.where(det.abs() > 1e-300, det, torch.ones_like(det))
        du = (f.real * fv.imag - fv.real * f.imag) / det
        dv = (fu.real * f.imag - f.real * fu.imag) / det
        u = (u - du).clamp(0.0, 1.0)
        v = (v - dv).clamp(0.0, 1.0)
    return u, v


def _spectral_refine(grid, psi, positions, iterations=6):
    """Newton on the trigonometric interpolant of psi; cost ~ count * N^2 per iteration."""
    Lx, Ly = grid.length
    coeff = grid.fft(psi.to(torch.complex128)) / grid.size
    kx, ky = grid.k64[0].reshape(-1), grid.k64[1].reshape(-1)
    coeff = coeff * torch.exp(1j * kx * Lx / 2)[:, None] * torch.exp(1j * ky * Ly / 2)[None, :]
    p = positions.clone()
    for _ in range(iterations):
        ex = torch.exp(1j * p[:, :1] * kx[None, :])          # (n, Nx)
        ey = torch.exp(1j * p[:, 1:] * ky[None, :])          # (n, Ny)
        f = ((ex @ coeff) * ey).sum(1)
        fx = (((ex * (1j * kx)) @ coeff) * ey).sum(1)
        fy = ((ex @ coeff) * ey * (1j * ky)).sum(1)
        det = fx.real * fy.imag - fy.real * fx.imag
        dx = (f.real * fy.imag - fy.real * f.imag) / det
        dy = (fx.real * f.imag - f.real * fx.imag) / det
        p = p - torch.stack([dx, dy], dim=1)
    return p


def find_vortices(grid, psi, *, min_density=0.0, refine=False):
    """Locate the vortices of a single 2D field.

    Plaquettes with non-zero phase winding hold a vortex; its position is the zero
    of the bilinear interpolant inside the plaquette (error ~dx^2 / core size).
    ``refine=True`` polishes positions by Newton iteration on the spectral
    interpolant (accurate to round-off for resolved fields, but costs
    ~count * N^2 per iteration, so use it for few vortices). A plaquette counts
    only if at least three of its four corner densities reach ``min_density``,
    which rejects phase noise outside a trapped cloud (including plaquettes that
    straddle its edge) while keeping real vortices: only one corner can lie very
    close to a vortex zero, and the second-closest is >= dx/2 away, with density
    ~ (dx / 2)^2 / (2 xi^2) of the bulk. A threshold of ~1e-4 of the bulk density
    is safe for dx <= xi. (Requiring all four corners dropped real vortices; a
    mean-density test let edge noise through.) Returns (positions, charges): float64 (n, 2) tensor of (x, y)
    wrapped into the box, and int64 (n,) charges.
    """
    if grid.ndim != 2 or psi.dim() != 2:
        raise ValueError("find_vortices needs a single 2D field")
    if not bool(torch.isfinite(psi).all()):
        raise ValueError("field contains non-finite values")
    winding = _winding(psi)
    if min_density > 0:
        winding = torch.where(_density_mask(psi, (-2, -1), min_density), winding,
                              torch.zeros_like(winding))
    idx = torch.nonzero(winding)
    i, j = idx[:, 0], idx[:, 1]
    charges = winding[i, j]
    nx, ny = grid.shape
    a = psi[i, j]
    b = psi[(i + 1) % nx, j]
    c = psi[(i + 1) % nx, (j + 1) % ny]
    d = psi[i, (j + 1) % ny]
    u, v = _bilinear_zero(a, b, c, d)
    x0, y0 = grid.x64[0].reshape(-1), grid.x64[1].reshape(-1)
    positions = torch.stack([x0[i] + u * grid.dx[0], y0[j] + v * grid.dx[1]], dim=1)
    if refine and len(positions):
        positions = _spectral_refine(grid, psi, positions)
    lengths = torch.tensor(grid.length, dtype=torch.float64, device=positions.device)
    positions = torch.remainder(positions + lengths / 2, lengths) - lengths / 2
    return positions, charges


def _density_mask(psi, dims, min_density):
    """True where at least three of the four plaquette corners reach min_density."""
    rho = psi.real ** 2 + psi.imag ** 2
    corners = torch.stack([rho, torch.roll(rho, -1, dims[0]), torch.roll(rho, -1, dims[1]),
                           torch.roll(rho, (-1, -1), dims)])
    return torch.sort(corners, dim=0).values[1] >= min_density


def find_vortex_points(grid, psi, *, min_density=0.0):
    """Points where vortex lines pierce the plaquettes of a single 3D field.

    For each axis c, plaquettes in the plane of the other two axes (a, b) with
    non-zero winding are pierced by a line running along +-c; the point is the
    bilinear zero inside the plaquette, at the plaquette's c coordinate. The
    winding sign gives the orientation: +1 means the line's tangent has a
    positive c component (circulation counterclockwise in (a, b), with (a, b, c)
    cyclic). ``min_density`` works as in :func:`find_vortices`.
    Returns (points, axes, charges): float64 (n, 3) positions wrapped into the box,
    int64 (n,) pierced-plane normal axis c, int64 (n,) charges.
    """
    if grid.ndim != 3 or psi.dim() != 3:
        raise ValueError("find_vortex_points needs a single 3D field")
    if not bool(torch.isfinite(psi).all()):
        raise ValueError("field contains non-finite values")
    lengths = torch.tensor(grid.length, dtype=torch.float64, device=psi.device)
    out_p, out_a, out_q = [], [], []
    for c, (a_ax, b_ax) in ((2, (0, 1)), (0, (1, 2)), (1, (2, 0))):
        dims = (a_ax, b_ax)
        winding = _winding(psi, dims)
        if min_density > 0:
            winding = torch.where(_density_mask(psi, dims, min_density), winding,
                                  torch.zeros_like(winding))
        idx = torch.nonzero(winding)
        if len(idx) == 0:
            continue
        shape = grid.shape

        def corner(da, db):
            j = idx.clone()
            j[:, a_ax] = (j[:, a_ax] + da) % shape[a_ax]
            j[:, b_ax] = (j[:, b_ax] + db) % shape[b_ax]
            return psi[j[:, 0], j[:, 1], j[:, 2]]

        u, v = _bilinear_zero(corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1))
        coords = [grid.x64[i].reshape(-1)[idx[:, i]] for i in range(3)]
        coords[a_ax] = coords[a_ax] + u * grid.dx[a_ax]
        coords[b_ax] = coords[b_ax] + v * grid.dx[b_ax]
        out_p.append(torch.stack(coords, dim=1))
        out_a.append(torch.full((len(idx),), c, dtype=torch.int64, device=psi.device))
        out_q.append(winding[idx[:, 0], idx[:, 1], idx[:, 2]])
    if not out_p:
        empty = torch.empty(0, dtype=torch.int64, device=psi.device)
        return torch.empty((0, 3), dtype=torch.float64, device=psi.device), empty, empty
    points = torch.cat(out_p)
    points = torch.remainder(points + lengths / 2, lengths) - lengths / 2
    return points, torch.cat(out_a), torch.cat(out_q)


def point_vortex_velocities(grid, positions, charges, images=None):
    """Velocities of point vortices (circulation 2 pi q, hbar = m = 1) in the
    doubly periodic box of ``grid``, consistent with :func:`dipgpe.fields.vortex_phase`.

    The phase is theta = Im log F + linear term with
    d(log F)/dz = (pi / Lx) sum_j q_j sum_n cot(pi (z - z_j - i n Ly) / Lx);
    the velocity grad theta at vortex i excludes its own n = 0 term.
    Returns a float64 (n, 2) tensor.
    """
    Lx, Ly = grid.length
    if images is None:
        images = math.ceil(math.log(1e12) / (2 * math.pi) * Lx / Ly)
    z = [complex(x, y) for x, y in positions]
    charges = [int(q) for q in charges]
    uniform_y = -2 * math.pi * sum(q * p[0] for p, q in zip(positions, charges)) / (Lx * Ly)
    out = []
    for i, zi in enumerate(z):
        G = 0j
        for j, (zj, q) in enumerate(zip(z, charges)):
            for n in range(-images, images + 1):
                if i == j and n == 0:
                    continue
                w = math.pi * (zi - zj - 1j * n * Ly) / Lx
                G += q * (math.pi / Lx) * (torch.tensor(w, dtype=torch.complex128).cos()
                                           / torch.tensor(w, dtype=torch.complex128).sin()).item()
        out.append((G.imag, G.real + uniform_y))
    return torch.tensor(out, dtype=torch.float64)


def line_components(grid, points, max_gap=None):
    """Label connected vortex lines: piercing points closer than ``max_gap``
    (default 1.5 * sqrt(3) * max dx, nearest periodic image) share a label.
    Returns (labels, count); labels are int64 in [0, count). Cost ~ n^2 memory,
    fine for a few thousand points.
    """
    n = len(points)
    if n == 0:
        return torch.empty(0, dtype=torch.int64), 0
    gap = max_gap or 1.5 * math.sqrt(3) * max(grid.dx)
    lengths = torch.tensor(grid.length, dtype=points.dtype, device=points.device)
    d = torch.remainder(points[:, None, :] - points[None, :, :] + lengths / 2, lengths) - lengths / 2
    adjacent = d.pow(2).sum(-1) <= gap * gap
    labels = torch.arange(n, device=points.device)
    while True:                                     # min-label propagation to a fixed point
        neighbour_min = torch.where(adjacent, labels[None, :], n).min(dim=1).values
        new = torch.minimum(labels, neighbour_min)
        new = new[new]                              # pointer jumping
        if torch.equal(new, labels):
            break
        labels = new
    unique, labels = torch.unique(labels, return_inverse=True)
    return labels.cpu(), len(unique)

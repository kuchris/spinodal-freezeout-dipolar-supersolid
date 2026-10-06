"""Initial fields and simple field operations."""

from __future__ import annotations

import math

import torch

from .diagnostics import norm


def _per_axis(value, ndim):
    return (float(value),) * ndim if isinstance(value, (int, float)) else tuple(float(v) for v in value)


def gaussian(grid, center=0.0, width=1.0, momentum=0.0):
    """Normalized Gaussian with |psi|^2 of standard deviation ``width`` per axis.

    psi = prod_i (2 pi s^2)^(-1/4) exp(-(x_i - c_i)^2 / (4 s^2) + i k_i x_i)
    """
    centers = _per_axis(center, grid.ndim)
    momenta = _per_axis(momentum, grid.ndim)
    s = float(width)
    exponent = 0.0
    for x, c, k in zip(grid.x64, centers, momenta):
        exponent = exponent + (-(x - c) ** 2 / (4 * s * s) + 1j * k * x)
    psi = (2 * math.pi * s * s) ** (-grid.ndim / 4) * torch.exp(exponent)
    return psi.to(grid.dtype)


def normalize(grid, psi, N=1.0):
    """Scale psi so that its norm is N (per leading index)."""
    scale = torch.sqrt(N / norm(grid, psi)).to(grid.real_dtype)
    return psi * scale.reshape(scale.shape + (1,) * grid.ndim)


def resample(psi, grid_from, grid_to):
    """psi on grid_from, Fourier-interpolated to grid_to (same box lengths, any
    point counts and precisions): the retained modes are copied, missing ones
    are zero. Exact for fields band-limited to the coarser grid; the Nyquist
    mode of an even coarse axis is split between +-k_N so the result stays
    consistent with real-valued interpolation. Leading axes are kept."""
    if any(abs(a - b) > 1e-12 * max(a, b) for a, b in zip(grid_from.length, grid_to.length)):
        raise ValueError("resample needs grids with the same box lengths")
    nd = grid_from.ndim
    spec = torch.fft.fftn(psi.to(torch.complex128), dim=grid_from.dims)
    out = torch.zeros(psi.shape[:psi.dim() - nd] + tuple(grid_to.shape), dtype=torch.complex128,
                      device=grid_to.device)
    spec = spec.to(grid_to.device)
    for axis in range(nd):
        n_from, n_to = grid_from.shape[axis], grid_to.shape[axis]
        n = min(n_from, n_to)
        if n_from == n_to:
            continue
        dim = psi.dim() - nd + axis
        pos, neg = (n + 1) // 2, n // 2                  # kept modes: 0..pos-1 and -neg..-1
        keep = torch.cat([spec.narrow(dim, 0, pos), spec.narrow(dim, spec.shape[dim] - neg, neg)], dim)
        if n % 2 == 0 and n_from < n_to:                  # coarse Nyquist mode: split between +-k_N
            nyq = keep.narrow(dim, pos, 1) * 0.5
            shape = list(keep.shape)
            shape[dim] = n_to
            full = torch.zeros(shape, dtype=keep.dtype, device=keep.device)
            full.narrow(dim, 0, pos).copy_(keep.narrow(dim, 0, pos))
            full.narrow(dim, pos, 1).copy_(nyq)
            full.narrow(dim, n_to - neg, neg).copy_(keep.narrow(dim, pos, neg))
            full.narrow(dim, n_to - neg, 1).copy_(nyq)
            spec = full
        elif n_from < n_to:
            shape = list(keep.shape)
            shape[dim] = n_to
            full = torch.zeros(shape, dtype=keep.dtype, device=keep.device)
            full.narrow(dim, 0, pos).copy_(keep.narrow(dim, 0, pos))
            full.narrow(dim, n_to - neg, neg).copy_(keep.narrow(dim, pos, neg))
            spec = full
        else:
            if n % 2 == 0:                               # fine -> coarse: +k_N aliases onto the coarse Nyquist
                keep = keep.clone()
                keep.narrow(dim, pos, 1).add_(spec.narrow(dim, pos, 1))
            spec = keep
    out.copy_(spec)
    scale = grid_to.size / grid_from.size
    return (torch.fft.ifftn(out, dim=grid_to.dims) * scale).to(grid_to.dtype)


def translate(grid, psi, displacement):
    """Exact periodic translation psi(x - d) via a k-space phase."""
    shifts = _per_axis(displacement, grid.ndim)
    phase = 0.0
    for k, d in zip(grid.k64, shifts):
        phase = phase + k * d
    return grid.ifft(grid.fft(psi) * torch.exp(-1j * phase).to(grid.dtype))


def vortex_phase(grid, positions, charges, images=None):
    """exp(i theta) for point vortices in a doubly periodic 2D box (net charge 0).

    theta = sum_j q_j sum_{|n| <= images} arg sin(pi (z - z_j - i n Ly) / Lx)
            - 2 pi (sum_j q_j x_j) y / (Lx Ly),           z = x + i y.
    Each factor is a row of vortices periodic in x; summing rows over n and the
    linear term make the total periodic in y as well. The truncation error decays
    like exp(-2 pi images Ly / Lx); by default images is chosen for ~1e-12
    (complex128 grids) or ~1e-8 (complex64). The phase winds by q_j around
    (x_j, y_j).

    Computed in the grid's precision with sin(A + iB) = sin A cosh B + i cos A sinh B,
    where A depends only on x and B only on y, so transcendental functions are
    evaluated on 1D arrays (complex128 on consumer GPUs is ~60x slower).
    """
    if grid.ndim != 2:
        raise ValueError("vortex_phase needs a 2D grid")
    charges = [int(q) for q in charges]
    if sum(charges) != 0:
        raise ValueError("net vortex charge must be zero in a periodic box")
    Lx, Ly = grid.length
    tolerance = 1e-12 if grid.dtype == torch.complex128 else 1e-8
    if images is None:
        images = math.ceil(-math.log(tolerance) / (2 * math.pi) * Lx / Ly)
    x, y = grid.x64[0], grid.x64[1]
    real = grid.real_dtype
    phase = torch.ones(grid.shape, dtype=grid.dtype, device=grid.device)
    for (xj, yj), q in zip(positions, charges):
        A = math.pi * (x - xj) / Lx
        sin_a, cos_a = torch.sin(A).to(real), torch.cos(A).to(real)
        vortex = torch.ones(grid.shape, dtype=grid.dtype, device=grid.device)
        for n in range(-images, images + 1):
            B = math.pi * (y - yj - n * Ly) / Lx
            factor = torch.complex(sin_a * torch.cosh(B).to(real), cos_a * torch.sinh(B).to(real))
            size = factor.abs()
            # A vortex exactly on a grid point: the phase is undefined there and the
            # amplitude is zero, so use 1 instead of 0/0.
            vortex = vortex * torch.where(size > 0, factor / torch.where(size > 0, size, 1.0), 1.0)
        vortex = vortex if q > 0 else vortex.conj()
        phase = phase * vortex ** abs(q)
        phase = phase / phase.abs()      # keep |phase| = 1 against rounding drift
    linear = -2 * math.pi * sum(q * p[0] for p, q in zip(positions, charges)) / (Lx * Ly)
    return phase * torch.exp(1j * linear * y).to(grid.dtype)


def vortex_amplitude(grid, positions, healing_length):
    """prod_j r_j / sqrt(r_j^2 + 2 xi^2): the usual vortex-core ansatz.

    r_j is a smooth periodic distance, (L/pi) sin(pi d / L) per axis, which equals
    the true distance near the core and keeps the field smooth across the box.
    Per-axis terms are computed on 1D arrays in float64, the product in the
    grid's precision.
    """
    amplitude = torch.ones(grid.shape, dtype=grid.real_dtype, device=grid.device)
    for p in positions:
        r2 = 0.0
        for x, L, c in zip(grid.x64, grid.length, p):
            r2 = r2 + ((L / math.pi * torch.sin(math.pi * (x - c) / L)) ** 2).to(grid.real_dtype)
        amplitude = amplitude * torch.sqrt(r2 / (r2 + 2 * healing_length ** 2))
    return amplitude


def imprint_vortices(grid, psi, positions, charges, healing_length, images=None):
    """psi times the vortex amplitude and phase of the given vortices."""
    return (psi * vortex_amplitude(grid, positions, healing_length)
            * vortex_phase(grid, positions, charges, images))


def random_vortices(grid, count, min_separation, seed=0, max_tries=100_000):
    """``count`` random vortex positions (count/2 of each sign), pairwise at least
    ``min_separation`` apart (nearest periodic image). Returns (positions, charges)."""
    if count % 2:
        raise ValueError("count must be even for zero net charge")
    rng = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(grid.length, dtype=torch.float64)
    points = torch.empty((0, grid.ndim), dtype=torch.float64)
    tries = 0
    while len(points) < count:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place vortices; lower count or min_separation")
        p = (torch.rand(grid.ndim, generator=rng, dtype=torch.float64) - 0.5) * lengths
        d = torch.remainder(points - p + lengths / 2, lengths) - lengths / 2
        if len(points) == 0 or bool((d.pow(2).sum(1) >= min_separation ** 2).all()):
            points = torch.cat([points, p[None]])
    charges = torch.tensor([1, -1] * (count // 2))[torch.randperm(count, generator=rng)]
    return points.tolist(), charges.tolist()


def random_phase(grid, k_cut, amplitude, seed=0):
    """exp(i phi) with phi a Gaussian random field band-limited to |k| < k_cut and
    standard deviation ``amplitude``; a common seed for vortex turbulence."""
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(grid.shape, generator=rng, dtype=torch.float64).to(grid.device)
    k2 = sum(k * k for k in grid.k64)
    phi = torch.fft.ifftn(torch.fft.fftn(noise) * (k2 < k_cut ** 2), dim=grid.dims).real
    phi = phi * (amplitude / phi.std())
    return torch.exp(1j * phi).to(grid.dtype)


def vortex_ring_phase(grid, center, radius, axis=2, sign=1, core_width=None):
    """exp(i phi) of a vortex ring in a triply periodic 3D box.

    The ring lies in the plane normal to ``axis`` through ``center``; ``sign`` = +1
    puts the flow through the ring along +axis, so the ring moves along +axis.

    Construction (there is no closed form in a periodic box):
      1. vorticity 2 pi t_hat on a Gaussian tube of width ``core_width``
         (default: one grid spacing) around the ring;
      2. velocity v = curl(laplacian^-1 vorticity) by real FFTs in the grid's
         precision (one component at a time, so 512^3 fits), plus a uniform flow
         2 pi^2 R^2 / (L1 L2 L3) along the axis, which makes the phase change along
         any periodic line exactly 2 pi times its linking number with the ring;
      3. phi on the plane farthest from the ring from its in-plane (curl-free)
         velocity, then phi = that + the integral of v along the axis. Lines that
         cross the vorticity tube pick up a non-integer multiple of 2 pi; the
         excess is removed by a smooth step (width core_width) at the ring plane,
         which puts the remaining phase defect inside the vortex core.
    Outside the tube v is the exact potential flow of a thin ring. (Rounding the
    excess as a ramp along the whole line instead left a phase cut along the
    cylinder s = R, with 2e-3 of the field energy near Nyquist.)
    """
    if grid.ndim != 3:
        raise ValueError("vortex_ring_phase needs a 3D grid")
    n = axis
    a1, a2 = [i for i in range(3) if i != n]
    L, real = grid.length, grid.real_dtype
    sigma = core_width or max(grid.dx)
    d = [(torch.remainder(grid.x64[i] - center[i] + L[i] / 2, L[i]) - L[i] / 2).to(real) for i in range(3)]
    s = torch.sqrt(d[a1] ** 2 + d[a2] ** 2)
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    tube = torch.exp(-((s - radius) ** 2 + d[n] ** 2) / (2 * sigma ** 2)) * (sign / sigma ** 2)

    # Real FFTs over (a1, a2, n) halve the last listed axis n; work in the grid's
    # precision and build one velocity component at a time to bound memory.
    fdims = (a1, a2, n)
    sizes = [grid.shape[i] for i in fdims]
    k = [grid.k64[i].to(real) for i in range(3)]
    k[n] = (2 * math.pi * torch.fft.rfftfreq(grid.shape[n], d=grid.dx[n], dtype=torch.float64)
            ).reshape([-1 if i == n else 1 for i in range(3)]).to(real).to(grid.device)
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    inv_k2 = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, 1.0), 0.0)
    # vorticity = 2 pi t_hat * tube / (2 pi sigma^2) = (-d2/s, d1/s, 0) * tube
    A1 = torch.fft.rfftn(tube * (-d[a2] / s_safe), dim=fdims) * inv_k2   # A_a1
    A2 = torch.fft.rfftn(tube * (d[a1] / s_safe), dim=fdims) * inv_k2    # A_a2
    del tube, s, s_safe

    def inverse(field_hat):
        return torch.fft.irfftn(field_hat, s=sizes, dim=fdims)

    vn = inverse(1j * (k[a1] * A2 - k[a2] * A1))           # v_n = (curl A)_n
    vn = vn + sign * 2 * math.pi ** 2 * radius ** 2 / (L[0] * L[1] * L[2])
    z = grid.x64[n].reshape(-1)
    zb = int(torch.argmin(torch.abs(torch.remainder(z - center[n], L[n]) - L[n] / 2)))
    v1 = inverse(-1j * k[n] * A2).select(n, zb).to(torch.float64)   # (curl A)_a1 = -d_n A_a2
    v2 = inverse(1j * k[n] * A1).select(n, zb).to(torch.float64)    # (curl A)_a2 = d_n A_a1
    del A1, A2

    # Base plane (farthest from the ring): in-plane flow is curl-free there.
    K1 = grid.k64[a1].reshape(-1)[:, None]
    K2 = grid.k64[a2].reshape(-1)[None, :]
    kp2 = K1 ** 2 + K2 ** 2
    num = -1j * (K1 * torch.fft.fft2(v1) + K2 * torch.fft.fft2(v2))
    phi_base = torch.fft.ifft2(torch.where(kp2 > 0, num / torch.where(kp2 > 0, kp2, 1.0), 0.0)).real.to(real)

    # Integrate v_n along the axis starting from the base plane.
    total = vn.mean(dim=n, keepdim=True) * L[n]              # 2 pi * (smooth linking number)
    remainder = total - 2 * math.pi * torch.round(total / (2 * math.pi))
    kn = k[n]
    fk = torch.fft.rfft(vn - total / L[n], dim=n)
    del vn
    integral = torch.fft.irfft(torch.where(kn != 0, fk / (1j * torch.where(kn != 0, kn, 1.0)), 0.0),
                               n=grid.shape[n], dim=n)
    del fk
    integral = integral - integral.select(n, zb).unsqueeze(n)
    distance = torch.remainder(grid.x64[n] - z[zb], L[n]).to(real)
    # Lines through the vorticity tube carry a non-integer total; remove the excess
    # with a smooth step at the ring plane, so the leftover defect sits in the core
    # (where the amplitude vanishes) instead of spreading along the whole line.
    ring_distance = float(torch.remainder(torch.tensor(center[n] - float(z[zb]), dtype=torch.float64), L[n]))
    step = 0.5 * (1 + torch.erf((distance - ring_distance) / (math.sqrt(2) * sigma)))
    phi = phi_base.unsqueeze(n) + integral + total / L[n] * distance - remainder * step
    return torch.polar(torch.ones_like(phi), phi).to(grid.dtype)


def vortex_ring_amplitude(grid, center, radius, healing_length, axis=2):
    """Core ansatz d / sqrt(d^2 + 2 xi^2), d = distance to the ring (nearest image)."""
    n = axis
    a1, a2 = [i for i in range(3) if i != n]
    L = grid.length
    d = [torch.remainder(grid.x64[i] - center[i] + L[i] / 2, L[i]) - L[i] / 2 for i in range(3)]
    dist2 = (torch.sqrt(d[a1] ** 2 + d[a2] ** 2) - radius) ** 2 + d[n] ** 2
    return torch.sqrt(dist2 / (dist2 + 2 * healing_length ** 2)).to(grid.real_dtype)


def imprint_vortex_rings(grid, psi, rings, healing_length):
    """psi times the amplitude and phase of each ring; rings are dicts with keys
    center, radius and optionally axis (default 2) and sign (default +1)."""
    for ring in rings:
        axis, sign = ring.get("axis", 2), ring.get("sign", 1)
        psi = (psi * vortex_ring_amplitude(grid, ring["center"], ring["radius"], healing_length, axis)
               * vortex_ring_phase(grid, ring["center"], ring["radius"], axis, sign))
    return psi

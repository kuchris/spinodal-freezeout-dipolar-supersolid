"""Truncated-Wigner initial noise.

Each mode u_m (orthonormal) receives a complex Gaussian amplitude a_m with
<|a_m|^2> = 1 / (exp(eps_m / kT) - 1) + 1/2 (thermal plus half a quantum), or
1/2 only (``temperature=0``); modes are kept below a cutoff energy (default
2 kT, Kirkby et al., PRR 7 (2025), App. A, Eqs. (A1)-(A2)). The noise field is
sum_m a_m u_m, to be added to the condensate (or ground state).

Modes: plane waves exp(i k x) / sqrt(L) along one periodic ``axis``, k != 0,
eps = k^2 / 2 (free particle), times, on 2D/3D grids, products of harmonic-
oscillator eigenfunctions (frequencies ``omegas``) on the other axes with
energies omega (n + 1/2) measured from the transverse ground state.
"""

from __future__ import annotations

import math

import torch


def hermite_functions(x, n_max, omega=1.0):
    """phi_0 .. phi_n_max of the 1D oscillator of frequency omega at x (float64 rows)."""
    s = math.sqrt(omega)
    xi = x * s
    out = [torch.exp(-xi * xi / 2) * (s / math.pi) ** 0.25]
    if n_max >= 1:
        out.append(math.sqrt(2.0) * xi * out[0])
    for n in range(2, n_max + 1):
        out.append(math.sqrt(2.0 / n) * xi * out[n - 1] - math.sqrt((n - 1) / n) * out[n - 2])
    return out


def wigner_noise(grid, temperature, *, axis=-1, omegas=None, cutoff=None, k_max=None, samples=None,
                 seed=0, return_modes=False):
    """Noise field(s) sum_m a_m u_m (see the module docstring), shape grid.shape
    or (samples, *grid.shape). ``temperature`` is k_B T in the model's energy unit;
    0 gives quantum noise only, with modes |k| < k_max (required then).
    ``cutoff`` (energy) defaults to 2 kT. ``omegas`` are the transverse
    oscillator frequencies for the axes other than ``axis`` (default 1).
    Returns (noise, info) with info = {"modes", "mean_added_atoms"} (the
    expected number of atoms the noise adds, <sum |a_m|^2>)."""
    axis = axis % grid.ndim
    if temperature <= 0 and k_max is None:
        raise ValueError("quantum noise needs k_max")
    if cutoff is None:
        cutoff = 2.0 * temperature if temperature > 0 else math.inf
    L = grid.length[axis]
    k1 = 2 * math.pi / L
    others = [d for d in range(grid.ndim) if d != axis]
    omegas = [1.0] * len(others) if omegas is None else list(omegas)
    # transverse quantum numbers with energy below the cutoff
    n_max = [int(cutoff / w) if math.isfinite(cutoff) else 0 for w in omegas]
    trans = [()]
    for nm in n_max:
        trans = [t + (n,) for t in trans for n in range(nm + 1)]
    trans = [t for t in trans if sum(n * w for n, w in zip(t, omegas)) < cutoff]
    l_max = math.floor(math.sqrt(2 * cutoff) / k1) if math.isfinite(cutoff) else math.floor(k_max / k1)
    if k_max is not None:
        l_max = min(l_max, math.floor(k_max / k1 - 1e-12))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    B = 1 if samples is None else int(samples)
    device = grid.device
    x_axis = grid.x64[axis].reshape(-1)
    herm = [hermite_functions(grid.x64[d].reshape(-1), nm, w) for d, nm, w in zip(others, n_max, omegas)]
    noise = torch.zeros((B,) + grid.shape, dtype=torch.complex128, device=device)
    added = 0.0
    count = 0
    for t in trans:
        e_t = sum(n * w for n, w in zip(t, omegas))
        ls = [l for l in range(-l_max, l_max + 1) if l != 0 and e_t + 0.5 * (k1 * l) ** 2 < cutoff]
        if not ls:
            continue
        eps = torch.tensor([e_t + 0.5 * (k1 * l) ** 2 for l in ls], dtype=torch.float64)
        occ = (1 / torch.expm1(eps / temperature) if temperature > 0 else torch.zeros_like(eps)) + 0.5
        a = torch.complex(torch.randn((B, len(ls)), generator=gen, dtype=torch.float64),
                          torch.randn((B, len(ls)), generator=gen, dtype=torch.float64)) * torch.sqrt(occ / 2)
        a = a.to(device)
        k = torch.tensor([k1 * l for l in ls], dtype=torch.float64, device=device)
        waves = torch.exp(1j * k[:, None] * x_axis[None, :]) / math.sqrt(L)       # (modes, N_axis)
        line = a @ waves.to(torch.complex128)                                      # (B, N_axis)
        prof = None                                                                # transverse product
        for d, n, h in zip(others, t, herm):
            shape = [1] * grid.ndim
            shape[d] = -1
            f = h[n].reshape(shape)
            prof = f if prof is None else prof * f
        shape = [B] + [1] * grid.ndim
        shape[1 + axis] = -1
        term = line.reshape(shape)
        noise = noise + (term * prof if prof is not None else term)
        added += float(occ.sum())
        count += len(ls)
    noise = noise.to(grid.dtype)
    if samples is None:
        noise = noise[0]
    return noise, {"modes": count, "mean_added_atoms": added}

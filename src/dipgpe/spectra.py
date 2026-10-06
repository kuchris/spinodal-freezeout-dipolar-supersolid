"""Kinetic-energy decomposition and spectra (Nore, Abid & Brachet 1997).

With psi = sqrt(rho) exp(i phi) and spectral gradients,
    |grad psi|^2 = |Re(psi* grad psi)|^2 / rho + |Im(psi* grad psi)|^2 / rho
                 = |grad sqrt(rho)|^2       + |sqrt(rho) v|^2,
so the kinetic energy splits exactly into a quantum-pressure part and the
density-weighted velocity w = sqrt(rho) v = Im(psi* grad psi) / |psi|.
A Helmholtz projection in k-space splits w into incompressible (div-free) and
compressible (curl-free) parts; the k = 0 mean flow is counted as incompressible.
The projection uses wavenumbers with the Nyquist entry set to zero: projecting
with the stored (signed) Nyquist wavenumber breaks the Hermitian symmetry of
the real field w and lost ~1e-6 of the energy when taking the real part.
At grid points where psi is exactly zero both fields are set to zero, which
drops |grad psi|^2 there; this does not happen for generic vortex positions.
"""

from __future__ import annotations

import math

import torch


def _projection_wavenumbers(grid):
    out = []
    for n, k in zip(grid.shape, grid.k):
        k = k.clone()
        if n % 2 == 0:
            k.view(-1)[n // 2] = 0.0
        out.append(k)
    return out


def _gradient(grid, psi):
    psi_k = grid.fft(psi)
    return [grid.ifft(1j * k * psi_k) for k in grid.k]


def velocity_fields(grid, psi):
    """Real fields [ndim, *space]: 'incompressible', 'compressible', 'quantum'."""
    grad = _gradient(grid, psi)
    amplitude = psi.abs()
    safe = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    zero = amplitude == 0
    w = torch.stack([torch.where(zero, 0.0, (psi.conj() * g).imag / safe) for g in grad])
    q = torch.stack([torch.where(zero, 0.0, (psi.conj() * g).real / safe) for g in grad])
    w_k = grid.fft(w)
    kp = _projection_wavenumbers(grid)
    k2 = sum(k * k for k in kp)
    k2_safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
    divergence = sum(k * w_k[i] for i, k in enumerate(kp))
    wc_k = torch.stack([k * divergence / k2_safe for k in kp])
    wc_k = torch.where(k2 > 0, wc_k, torch.zeros_like(wc_k))
    return {"incompressible": grid.ifft(w_k - wc_k).real,
            "compressible": grid.ifft(wc_k).real,
            "quantum": q}


def kinetic_energies(grid, psi):
    """Dict of floats: incompressible, compressible, quantum and total kinetic energy."""
    fields = velocity_fields(grid, psi)
    out = {name: float(0.5 * grid.integrate((f * f).sum(0))) for name, f in fields.items()}
    out["total"] = float(0.5 * (grid.k2 * grid.fft(psi).abs() ** 2).sum(dtype=torch.float64)
                         * grid.dV / grid.size)
    return out


def energy_spectra(grid, psi):
    """Angle-integrated spectra E(k) of each component, normalized so that
    sum_k E(k) dk equals that component's energy.

    Shells have width dk = 2 pi / max(L); returns (k, dict of spectra) as float64
    tensors on the CPU.
    """
    fields = velocity_fields(grid, psi)
    dk = 2 * math.pi / max(grid.length)
    shell = torch.round(grid.k2.to(torch.float64).sqrt() / dk).to(torch.int64).reshape(-1)
    bins = int(shell.max()) + 1
    spectra = {}
    for name, f in fields.items():
        power = (grid.fft(f).abs() ** 2).sum(0).to(torch.float64).reshape(-1)
        e = torch.bincount(shell, weights=power, minlength=bins) * (0.5 * grid.dV / grid.size / dk)
        spectra[name] = e.cpu()
    return torch.arange(bins, dtype=torch.float64) * dk, spectra

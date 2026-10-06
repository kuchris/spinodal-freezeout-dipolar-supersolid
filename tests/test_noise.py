"""Truncated-Wigner noise: mode occupations, transverse modes, added atoms."""

import math

import torch

import dipgpe
from dipgpe.noise import hermite_functions, wigner_noise
from util import C128, DEVICE


def test_plane_wave_occupations_match_bose_einstein_plus_half():
    grid = dipgpe.Grid(256, 40.0, dtype=C128, device=DEVICE)
    kT, B = 3.0, 4000
    noise, info = wigner_noise(grid, kT, samples=B, seed=1)
    x = grid.x64[0]
    for l in (1, 3, -5, 12):
        k = 2 * math.pi * l / 40.0
        u = torch.exp(1j * k * x) / math.sqrt(40.0)
        a = grid.integrate(u.conj().to(C128) * noise)                  # (B,)
        occ = 1 / math.expm1(0.5 * k * k / kT) + 0.5
        mean = float((a.abs() ** 2).mean())
        assert abs(mean / occ - 1) < 4 / math.sqrt(B), (l, mean, occ)
    added = float(grid.integrate(noise.abs() ** 2).mean())
    assert abs(added / info["mean_added_atoms"] - 1) < 0.02
    # cutoff 2 kT: |k| < 2 sqrt(kT)
    assert info["modes"] == 2 * math.floor(2 * math.sqrt(kT) / (2 * math.pi / 40.0))


def test_quantum_noise_and_transverse_modes():
    grid = dipgpe.Grid((32, 32, 48), (10.0, 10.0, 30.0), dtype=C128, device=DEVICE)
    h = hermite_functions(grid.x64[0].reshape(-1), 4)
    dx = grid.dx[0]
    G = torch.stack(h) @ torch.stack(h).T * dx
    assert float((G.cpu() - torch.eye(5, dtype=torch.float64)).abs().max()) < 1e-6     # box L = 10 cuts the tails
    noise, info = wigner_noise(grid, 0.0, k_max=2.0, samples=500, seed=2)
    # temperature 0: transverse ground mode only (no energy cutoff), plane waves |k| < k_max, occupation 1/2
    l_max = math.floor(2.0 / (2 * math.pi / 30.0) - 1e-12)
    assert info["modes"] == 2 * l_max and abs(info["mean_added_atoms"] - l_max) < 1e-12
    added = float(grid.integrate(noise.abs() ** 2).mean())
    assert abs(added / info["mean_added_atoms"] - 1) < 0.05
    noise_t, info_t = wigner_noise(grid, 3.0, samples=2, seed=3)
    assert info_t["modes"] > info["modes"] and noise_t.shape == (2,) + grid.shape

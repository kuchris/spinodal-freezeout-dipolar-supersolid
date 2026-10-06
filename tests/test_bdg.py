"""Gate: Bogoliubov-de Gennes spectra against exact results."""

import math

import torch

import dipgpe
from dipgpe.bdg import bogoliubov
from util import C128, DEVICE


def uniform(grid, n):
    return torch.full(grid.shape, math.sqrt(n), dtype=C128, device=DEVICE)


def test_uniform_contact_gas_1d_with_bloch_shift():
    L, n, g = 20.0, 2.0, 3.0
    grid = dipgpe.Grid(256, L, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    for q in (0.0, 0.1):
        w, _, info = bogoliubov(model, uniform(grid, n), q=q, n_modes=5)
        ks = sorted(abs(2 * math.pi * j / L + q) for j in range(-4, 5))[:5]
        exact = [math.sqrt(k * k / 2 * (k * k / 2 + 2 * g * n)) for k in ks]
        assert max(abs(a - b) for a, b in zip(w.tolist(), exact)) < 1e-7, (q, w.tolist(), exact)


def test_deflation_removes_one_mode_of_a_degenerate_pair():
    """q = 0: the modes cos(k1 z), sin(k1 z) are degenerate (and L, M both keep the
    complement of cos(k1 z) invariant); deflating cos(k1 z) leaves omega(k1) once,
    followed by omega(k2) twice."""
    L, n, g = 20.0, 2.0, 3.0
    grid = dipgpe.Grid(256, L, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    k1 = 2 * math.pi / L
    cos1 = torch.cos(k1 * grid.x64[0]).to(C128)
    w, _, _ = bogoliubov(model, uniform(grid, n), n_modes=4, deflate=[cos1])
    om = lambda k: math.sqrt(k * k / 2 * (k * k / 2 + 2 * g * n))  # noqa: E731
    exact = [0.0, om(k1), om(2 * k1), om(2 * k1)]
    assert max(abs(a - b) for a, b in zip(w.tolist(), exact)) < 1e-6, (w.tolist(), exact)


def test_uniform_dipolar_lhy_gas_3d():
    """omega^2 = e_k (e_k + 2 n [g + U_dd(k) + (3/2) gamma n^(1/2)]), periodic kernel."""
    L, n, g, gdd, gamma = 12.0, 1.5, 2.0, 1.2, 0.3
    grid = dipgpe.Grid((24, 24, 24), L, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Contact(g), dipgpe.Dipolar(grid, gdd, direction=(0, 0, 1), cutoff=None), dipgpe.LHY(gamma))
    q = 0.07
    w, _, _ = bogoliubov(model, uniform(grid, n), q=q, axis=0, n_modes=4)
    exact = []
    k1 = 2 * math.pi / L
    for kx in (q, q - k1, q + k1):
        for ky, kz in ((0, 0), (k1, 0), (-k1, 0), (0, k1), (0, -k1)):
            k2 = kx * kx + ky * ky + kz * kz
            udd = gdd * (3 * kz * kz / k2 - 1)
            ek = k2 / 2
            exact.append(math.sqrt(ek * (ek + 2 * n * (g + udd + 1.5 * gamma * math.sqrt(n)))))
    exact = sorted(exact)[:4]
    assert max(abs(a - b) / b for a, b in zip(w.tolist(), exact)) < 1e-7, (w.tolist(), exact)


def test_trapped_kohn_and_breathing_modes_2d():
    grid = dipgpe.Grid((96, 96), 16.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(100.0))
    psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 2.0)))
    w, _, _ = bogoliubov(model, psi, n_modes=10)
    w = w.tolist()
    assert w[0] < 1e-5                                    # gauge mode
    assert abs(w[1] - 1) < 1e-6 and abs(w[2] - 1) < 1e-6  # Kohn (dipole) modes at the trap frequency
    assert abs(w[7] - 2) < 1e-6                           # 2D breathing mode at 2 omega (exact, any g),
    assert w[3] > 1.4 and abs(w[3] - w[4]) < 1e-6         # after the quadrupole (1.464) and octupole pairs

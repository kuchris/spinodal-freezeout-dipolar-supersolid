"""Gate: the ground-state minimizer finds E = d/2 for g = 0 and approaches Thomas-Fermi for large g."""

import math

import pytest

import dipgpe
from util import C64, C128, DEVICE


@pytest.mark.parametrize("ndim, N, L", [(1, 128, 16.0), (2, 64, 16.0), (3, 32, 12.0)])
@pytest.mark.parametrize("dtype, tol", [(C64, 2e-6), (C128, 1e-9)])
def test_harmonic_ground_state_energy(ndim, N, L, dtype, tol):
    grid = dipgpe.Grid((N,) * ndim, L, dtype=dtype, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)))
    psi0 = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.3, 1.3))
    psi, _ = dipgpe.ground_state(model, psi0)
    assert abs(float(dipgpe.energy(model, psi)) - ndim / 2) < tol


def thomas_fermi_1d(g):
    """Ground state of a 1D trap (omega = 1, N = 1)."""
    mu_tf = 0.5 * (1.5 * g) ** (2 / 3)
    radius = math.sqrt(2 * mu_tf)
    L = 2 * radius + 12.0
    N = 1 << math.ceil(math.log2(L / min(0.05, 0.5 / math.sqrt(mu_tf))))
    grid = dipgpe.Grid(N, L, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(g))
    psi0 = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, radius / 2))
    psi, info = dipgpe.ground_state(model, psi0)
    kinetic, (trap, interaction) = dipgpe.energy_components(model, psi)
    # Virial theorem in d dimensions: 2 E_kin - 2 E_trap + d E_int = 0.
    virial = (2 * kinetic - 2 * trap + interaction) / dipgpe.energy(model, psi)
    return float(info["mu"]) / mu_tf - 1, float(virial)


def test_thomas_fermi_limit_1d():
    deviations = []
    for g in (10.0, 100.0, 1000.0, 10000.0):
        deviation, virial = thomas_fermi_1d(g)
        deviations.append(deviation)
        assert abs(virial) < 1e-4, (g, virial)
    assert all(a > b > 0 for a, b in zip(deviations, deviations[1:])), deviations
    assert deviations[-1] < 1e-4


def test_thomas_fermi_2d():
    g = 1000.0
    mu_tf = math.sqrt(g / math.pi)
    grid = dipgpe.Grid((256, 256), 24.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(g))
    psi0 = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 2.0))
    psi, info = dipgpe.ground_state(model, psi0)
    kinetic, (trap, interaction) = dipgpe.energy_components(model, psi)
    virial = float((2 * kinetic - 2 * trap + 2 * interaction) / dipgpe.energy(model, psi))
    assert abs(virial) < 1e-4
    assert 0 < float(info["mu"]) / mu_tf - 1 < 0.01

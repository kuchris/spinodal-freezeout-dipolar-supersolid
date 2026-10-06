"""Gates: Kohn dipole mode at the trap frequency and 2D breathing mode at 2 omega.

Both are exact consequences of the continuum GPE in a harmonic trap for any
contact interaction strength:
  Kohn:      <x>(t) = x0 cos(omega t) after displacing the ground state by x0.
  Breathing: d^2<r^2>/dt^2 = 4 E - 4 omega^2 <r^2> in 2D (E per particle), so
             <r^2>(t) = E + (<r^2>(0) - E) cos(2t) for omega = 1, N = 1 and a
             real initial state.
"""

import math

import pytest

import dipgpe
from util import C64, C128, DEVICE


def interacting_ground_state(dtype, omega, g=100.0, N=128, L=16.0):
    grid = dipgpe.Grid((N, N), L, dtype=dtype, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid, omega)), dipgpe.Contact(g))
    psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5)))
    return grid, psi


@pytest.mark.parametrize("dtype, tol", [(C64, 1e-3), (C128, 1e-5)])
def test_kohn_mode(dtype, tol):
    grid, psi = interacting_ground_state(dtype, 1.0)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(100.0))
    x0, dt, chunks = 0.5, 0.002, 40
    psi = dipgpe.translate(grid, psi, (x0, 0.0))
    stepper = dipgpe.SplitStep(model, dt, keep_norm=dtype == C64)
    steps = round(4 * math.pi / dt) // chunks
    worst, t = 0.0, 0.0
    for _ in range(chunks):
        psi = stepper.evolve(psi, steps)
        t += steps * dt
        x, y = (float(c) for c in dipgpe.center_of_mass(grid, psi))
        worst = max(worst, abs(x - x0 * math.cos(t)), abs(y))
    assert worst / x0 < tol


@pytest.mark.parametrize("dtype, tol", [(C64, 1e-3), (C128, 5e-5)])
def test_breathing_mode_2d(dtype, tol):
    grid, psi = interacting_ground_state(dtype, 1.2)  # quench the trap 1.2 -> 1
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(100.0))
    r2 = grid.r2()
    E = float(dipgpe.energy(model, psi))
    amplitude = float(dipgpe.expectation(grid, psi, r2)) - E
    dt, chunks = 0.002, 40
    stepper = dipgpe.SplitStep(model, dt, keep_norm=dtype == C64)
    steps = round(2 * math.pi / dt) // chunks
    worst, t = 0.0, 0.0
    for _ in range(chunks):
        psi = stepper.evolve(psi, steps)
        t += steps * dt
        predicted = E + amplitude * math.cos(2 * t)
        worst = max(worst, abs(float(dipgpe.expectation(grid, psi, r2)) - predicted))
    assert worst / abs(amplitude) < tol

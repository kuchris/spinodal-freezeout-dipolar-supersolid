"""Gate: real time conserves the norm to round-off and the energy to splitting error."""

import pytest
import torch

import dipgpe
from util import C64, C128, DEVICE


def run(dtype, dt, T=10.0, device=DEVICE, keep_norm=False):
    """Max relative norm and energy deviation over a trapped nonlinear 2D run."""
    grid = dipgpe.Grid((64, 64), 16.0, dtype=dtype, device=device)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(10.0))
    psi = dipgpe.gaussian(grid, (0.7, -0.3), 0.8, (0.5, 0.2))
    stepper = dipgpe.SplitStep(model, dt, keep_norm=keep_norm)
    N0, E0 = float(dipgpe.norm(grid, psi)), float(dipgpe.energy(model, psi))
    steps, chunks = round(T / dt), 20
    dN = dE = 0.0
    for _ in range(chunks):
        psi = stepper.evolve(psi, steps // chunks)
        dN = max(dN, abs(float(dipgpe.norm(grid, psi)) / N0 - 1))
        dE = max(dE, abs(float(dipgpe.energy(model, psi)) / E0 - 1))
    return dN, dE, steps


def test_complex128_norm_and_energy():
    dN1, dE1, _ = run(C128, 0.01)
    dN2, dE2, _ = run(C128, 0.005)
    assert max(dN1, dN2) < 1e-11
    assert dE2 < 5e-6
    assert 3.5 < dE1 / dE2 < 4.5  # second-order energy error


def test_complex64_norm_drift_is_bounded_and_removable():
    """Single-precision cuFFT loses ~1e-7 to 4e-7 of the norm per FFT pair."""
    dN, dE, steps = run(C64, 0.01)
    assert dN / steps < 1e-6
    dN_kept, dE_kept, _ = run(C64, 0.01, keep_norm=True)
    assert dN_kept < 1e-6
    assert dE_kept < 1e-4

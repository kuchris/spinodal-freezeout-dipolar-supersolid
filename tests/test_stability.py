"""The split-step resonance instability and dipgpe.max_stable_dt.

On a uniform condensate (g n = 1) with a vortex dipole, a Strang step slightly
above the bound blows up within a few tens of time units, while dt =
max_stable_dt stays stable to T = 600. Linear problems never trigger it.
Long horizons matter: an instability that grows ~8x per 50 time units looks
stable over T = 150 (docs/ISSUES.md P2-11).
"""

import pytest
import torch

import dipgpe
from dipgpe.fields import imprint_vortices
from util import C128, DEVICE


def dipole_energy_drift(dt, T):
    grid = dipgpe.Grid((128, 128), 64.0, dtype=C128, device=DEVICE)
    psi = imprint_vortices(grid, torch.ones(grid.shape, dtype=C128, device=DEVICE),
                           [(0.1, 8.1), (0.1, -7.9)], [1, -1], 1.0)
    model = dipgpe.GPE(grid, dipgpe.Contact(1.0))
    stepper = dipgpe.SplitStep(model, dt)
    E0, worst = float(dipgpe.energy(model, psi)), 0.0
    while stepper.t < T:
        psi = stepper.evolve(psi, 100)
        worst = max(worst, abs(float(dipgpe.energy(model, psi)) / E0 - 1))
        if worst > 1e-2:
            break
    return worst


def test_max_stable_dt_value():
    grid = dipgpe.Grid((128, 128), 64.0, dtype=C128)
    k2_max = 2 * (torch.pi / 0.5) ** 2
    assert abs(dipgpe.max_stable_dt(grid, safety=1.0) - 2 * torch.pi / k2_max) < 1e-15
    assert dipgpe.max_stable_dt(grid, dipgpe.YOSHIDA4) < dipgpe.max_stable_dt(grid)


def test_above_limit_blows_up_and_warns():
    grid = dipgpe.Grid((128, 128), 64.0, dtype=C128)
    dt = 1.15 * dipgpe.max_stable_dt(grid, safety=1.0)
    with pytest.warns(RuntimeWarning, match="stability limit"):
        drift = dipole_energy_drift(dt, 60.0)
    assert drift > 1e-2


def test_at_max_stable_dt_stays_stable():
    grid = dipgpe.Grid((128, 128), 64.0, dtype=C128)
    assert dipole_energy_drift(dipgpe.max_stable_dt(grid), 600.0) < 1e-6


def test_linear_model_does_not_warn():
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C128)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dipgpe.SplitStep(dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid))), 1.0)

"""Gate: measured convergence order in dt is 2 (Strang) and 4 (Yoshida).

Runs in complex128 on a nonlinear trapped 2D problem; the reference uses the
same grid, so only the time-splitting error is measured. All dt stay below
dipgpe.max_stable_dt for this grid and divide T exactly.
"""

import math

import pytest

import dipgpe
from util import C128, DEVICE, rel_l2


def trapped_problem():
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(10.0))
    psi = dipgpe.gaussian(grid, (0.7, -0.3), 0.8, (0.5, 0.2))
    return model, psi


@pytest.mark.parametrize("scheme, dts, order", [
    (dipgpe.STRANG, (1 / 64, 1 / 128, 1 / 256), 2),
    (dipgpe.YOSHIDA4, (1 / 100, 1 / 200, 1 / 400), 4),
])
def test_convergence_order(scheme, dts, order):
    model, psi = trapped_problem()
    T = 1.0
    ref = dipgpe.SplitStep(model, 1e-3, dipgpe.YOSHIDA4).evolve(psi, 1000)
    errors = [rel_l2(dipgpe.SplitStep(model, dt, scheme).evolve(psi, round(T / dt)), ref) for dt in dts]
    measured = math.log2(errors[-2] / errors[-1])
    assert abs(measured - order) < 0.15, (errors, measured)


def test_yoshida_rejected_in_imaginary_time():
    model, _ = trapped_problem()
    try:
        dipgpe.SplitStep(model, 0.01, dipgpe.YOSHIDA4, imaginary=True)
    except ValueError:
        return
    raise AssertionError("negative substeps must be rejected in imaginary time")

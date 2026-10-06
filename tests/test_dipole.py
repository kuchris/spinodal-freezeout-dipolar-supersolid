"""Gate: a vortex dipole moves at the expected velocity.

Uniform condensate with g = n0 = 1 (xi = c = 1), L = 128, dipole separation d.
The velocity is the slope of the dipole's mean x position (refined vortex
positions) over the last three quarters of T = 4 d, at dt = max_stable_dt.

Checks:
  * resolution: dx = 0.5 xi and dx = 0.25 xi agree (measured 6e-6 relative);
  * physics: the deviation from the periodic point-vortex velocity (images
    included) shrinks as d / xi grows (measured +7.5%, +2.6%, +1.2% for
    d = 8, 16, 32): the finite-core correction vanishes for d >> xi.
"""

import math

import torch

import dipgpe
from dipgpe.fields import imprint_vortices
from dipgpe.vortices import find_vortices, point_vortex_velocities
from util import C128, DEVICE


def dipole_velocity(N, d, L=128.0, chunks=40):
    grid = dipgpe.Grid((N, N), L, dtype=C128, device=DEVICE)
    positions, charges = [(0.13, d / 2 + 0.07), (0.13, -d / 2 + 0.07)], [1, -1]
    psi = imprint_vortices(grid, torch.ones(grid.shape, dtype=C128, device=DEVICE),
                           positions, charges, 1.0)
    T = 4 * d
    n = math.ceil(T / chunks / dipgpe.max_stable_dt(grid))
    stepper = dipgpe.SplitStep(dipgpe.GPE(grid, dipgpe.Contact(1.0)), T / chunks / n)
    times, xs = [], []
    for _ in range(chunks):
        psi = stepper.evolve(psi, n)
        found, _ = find_vortices(grid, psi, refine=True)
        assert len(found) == 2
        times.append(stepper.t)
        xs.append(float(found[:, 0].mean()))
    t, x = torch.tensor(times), torch.tensor(xs)
    late = t >= T / 4
    A = torch.stack([t[late], torch.ones(int(late.sum()), dtype=t.dtype)], 1)
    v = torch.linalg.lstsq(A, x[late, None]).solution[0, 0].item()
    v_point = point_vortex_velocities(grid, positions, charges)[:, 0].mean().item()
    return v, v_point


def test_dipole_velocity_is_resolution_converged():
    v_coarse, _ = dipole_velocity(256, 16.0)
    v_fine, _ = dipole_velocity(512, 16.0)
    assert abs(v_coarse / v_fine - 1) < 1e-4


def test_dipole_approaches_point_vortex_velocity():
    deviations = []
    for d in (8.0, 16.0, 32.0):
        v, v_point = dipole_velocity(256, d)
        deviations.append(v / v_point - 1)
    assert all(a > b > 0 for a, b in zip(deviations, deviations[1:])), deviations
    assert deviations[-1] < 0.02

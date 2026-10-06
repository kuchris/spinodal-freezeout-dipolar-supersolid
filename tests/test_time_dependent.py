"""Time-dependent potentials.

A harmonic trap whose centre moves as a sin(W t) drives the centre of mass
exactly like a forced oscillator (Kohn), for any interaction strength:
    x(t) = a / (1 - W^2) * (sin(W t) - W sin(t))   from rest at the origin.
"""

import math

import pytest
import torch

import dipgpe
from util import C128, DEVICE


def moving_trap_error(dt, a=0.3, W=0.5, T=10.0):
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C128, device=DEVICE)
    static = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(50.0))
    psi, _ = dipgpe.ground_state(static, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.2)))
    trap = dipgpe.Potential(lambda t: dipgpe.harmonic(grid, center=(a * math.sin(W * t), 0.0)))
    stepper = dipgpe.SplitStep(dipgpe.GPE(grid, trap, dipgpe.Contact(50.0)), dt)
    worst, chunks = 0.0, 20
    for _ in range(chunks):
        psi = stepper.evolve(psi, round(T / dt) // chunks)
        t = stepper.t
        exact = a / (1 - W * W) * (math.sin(W * t) - W * math.sin(t))
        x, y = (float(c) for c in dipgpe.center_of_mass(grid, psi))
        worst = max(worst, abs(x - exact), abs(y))
    assert abs(stepper.t - T) < 1e-9
    return worst


def test_moving_trap_follows_forced_oscillator_at_second_order():
    e1, e2 = moving_trap_error(0.01), moving_trap_error(0.005)
    assert e2 < 1e-5
    assert 3.5 < e1 / e2 < 4.5, (e1, e2)


def test_gaussian_obstacle_moves_and_wraps():
    grid = dipgpe.Grid((64, 32), (16.0, 8.0), dtype=C128)
    V = dipgpe.gaussian_obstacle(grid, 2.0, 1.0, lambda t: (t, 0.0))
    peak = lambda v: tuple(int(i) for i in torch.nonzero(v == v.max())[0])
    assert peak(V(0.0)) == (32, 16)
    assert peak(V(8.0)) == (0, 16)  # x = 8 wraps to the box edge -8
    assert abs(float(V(0.0).max()) - 2.0) < 1e-12


def test_constant_callable_strengths_match_numbers():
    grid = dipgpe.Grid((48, 48), 12.0, dtype=C128, device=DEVICE)
    psi0 = dipgpe.normalize(grid, dipgpe.translate(grid, dipgpe.gaussian(grid, 0.0, 1.2), (0.5, 0.0)), 5.0)
    out = []
    for g, gamma in ((40.0, 3.0), (lambda t: 40.0, lambda t: 3.0)):
        model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(g), dipgpe.LHY(gamma))
        out.append(dipgpe.SplitStep(model, 0.005).evolve(psi0, 200))
    assert float((out[0] - out[1]).abs().max()) < 1e-13


def ramp_run(dt, T=2.0, g0=20.0, g1=80.0, gamma0=1.0, gamma1=4.0):
    """1D trap, g and gamma ramped linearly over T. Returns the final field and
    the energy balance error |E(T) - E(0) - int dH/dt dt| / |E(0)|, with
    dH/dt = g' int rho^2 / 2 + gamma' (2/5) int rho^(5/2) (trapezoid on the chunks)."""
    grid = dipgpe.Grid(256, 24.0, dtype=C128, device=DEVICE)
    g = lambda t: g0 + (g1 - g0) * t / T
    gamma = lambda t: gamma0 + (gamma1 - gamma0) * t / T
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(g), dipgpe.LHY(gamma))
    psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5), 3.0), t=0.0)
    stepper = dipgpe.SplitStep(model, dt)
    rate = lambda f: float(0.5 * (g1 - g0) / T * grid.integrate(f.abs() ** 4)
                           + 0.4 * (gamma1 - gamma0) / T * grid.integrate(f.abs() ** 5))
    E0 = float(dipgpe.energy(model, psi, 0.0))
    work, prev, chunks = 0.0, rate(psi), 40
    for _ in range(chunks):
        psi = stepper.evolve(psi, round(T / dt) // chunks)
        now = rate(psi)
        work += 0.5 * (prev + now) * T / chunks
        prev = now
    return psi, abs(float(dipgpe.energy(model, psi, T)) - E0 - work) / abs(E0)


def test_interaction_ramp_energy_balance_and_second_order():
    psi1, balance = ramp_run(2e-3)
    assert balance < 2e-4                                  # dominated by the trapezoid on 40 chunks
    psi2, _ = ramp_run(1e-3)
    ref, _ = ramp_run(2.5e-4)
    ratio = float((psi1 - ref).abs().max() / (psi2 - ref).abs().max())
    assert 3.3 < ratio < 4.7, ratio                        # second order with the ramp


def test_ground_state_and_energy_use_the_time_argument():
    grid = dipgpe.Grid(128, 16.0, dtype=C128, device=DEVICE)
    ramped = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(lambda t: 10.0 + 5.0 * t))
    fixed = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0))
    seed = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.0), 2.0)
    a, _ = dipgpe.ground_state(ramped, seed, t=2.0)
    b, _ = dipgpe.ground_state(fixed, seed)
    assert abs(float(dipgpe.energy(ramped, a, 2.0)) - float(dipgpe.energy(fixed, b))) < 1e-10
    assert ramped.terms[1].g == 10.0

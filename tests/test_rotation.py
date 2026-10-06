"""Rotating frame (-Omega L_z): exact checks of the split rotation step.

* An angular-momentum eigenstate (x +- i y) exp(-r^2/2) of the isotropic trap has
  rotating-frame energy 2 -+ Omega, so its phase turns at exactly that rate.
* Kohn's theorem in the rotating frame: a displaced cloud's centre of mass is the
  lab-frame solution x0 cos t rotated by -Omega t, for any interaction.
* Strang / Yoshida orders and energy conservation with rotation.
"""

import math

import pytest
import torch

import dipgpe
from util import C128, DEVICE, rel_l2


def trap_grid(N=64, L=16.0):
    return dipgpe.Grid((N, N), L, dtype=C128, device=DEVICE)


@pytest.mark.parametrize("m", [1, -1])
def test_angular_momentum_eigenstate_phase(m):
    grid, omega, T = trap_grid(), 0.4, 3.0
    x, y = grid.x64
    psi0 = ((x + 1j * m * y) * torch.exp(-(x * x + y * y) / 2) / math.sqrt(math.pi)).to(C128)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Rotation(omega))
    E = 2 - m * omega
    assert abs(float(dipgpe.angular_momentum(grid, psi0)) - m) < 1e-12
    assert abs(float(dipgpe.energy(model, psi0)) - E) < 1e-12
    out = dipgpe.SplitStep(model, 1e-3).evolve(psi0, round(T / 1e-3))
    assert rel_l2(out, psi0 * complex(math.cos(E * T), -math.sin(E * T))) < 1e-5


def test_kohn_mode_seen_from_the_rotating_frame():
    grid, omega, x0 = trap_grid(96, 20.0), 0.3, 0.8
    static = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0))
    psi, _ = dipgpe.ground_state(static, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.2)))
    psi = dipgpe.translate(grid, psi, (x0, 0.0))
    rotating = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0), dipgpe.Rotation(omega))
    stepper = dipgpe.SplitStep(rotating, 0.002)
    worst = 0.0
    for _ in range(20):
        psi = stepper.evolve(psi, 250)
        t = stepper.t
        lab = x0 * math.cos(t)
        expected = (lab * math.cos(omega * t), -lab * math.sin(omega * t))
        com = dipgpe.center_of_mass(grid, psi)
        worst = max(worst, abs(float(com[0]) - expected[0]), abs(float(com[1]) - expected[1]))
    assert worst < 1e-5


def rotating_problem():
    grid = trap_grid()
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(10.0), dipgpe.Rotation(0.5))
    return model, dipgpe.gaussian(grid, (0.7, -0.3), 0.8, (0.5, 0.2))


@pytest.mark.parametrize("scheme, dts, order", [
    (dipgpe.STRANG, (1 / 64, 1 / 128, 1 / 256), 2),
    (dipgpe.YOSHIDA4, (1 / 100, 1 / 200, 1 / 400), 4),
])
def test_convergence_order_with_rotation(scheme, dts, order):
    model, psi = rotating_problem()
    ref = dipgpe.SplitStep(model, 1e-3, dipgpe.YOSHIDA4).evolve(psi, 1000)
    errors = [rel_l2(dipgpe.SplitStep(model, dt, scheme).evolve(psi, round(1 / dt)), ref) for dt in dts]
    assert abs(math.log2(errors[-2] / errors[-1]) - order) < 0.15, errors


def test_rotating_frame_energy_is_conserved():
    model, psi = rotating_problem()
    E0 = float(dipgpe.energy(model, psi))
    stepper = dipgpe.SplitStep(model, 0.005)
    worst = 0.0
    for _ in range(20):
        psi = stepper.evolve(psi, 100)
        worst = max(worst, abs(float(dipgpe.energy(model, psi)) / E0 - 1))
    assert worst < 1e-5


def test_rotation_rejected_in_1d():
    with pytest.raises(ValueError):
        dipgpe.GPE(dipgpe.Grid(64, 16.0), dipgpe.Rotation(0.5))

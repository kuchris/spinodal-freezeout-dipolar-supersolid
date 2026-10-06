"""Gate: 1D bright and dark solitons propagate with the exact profile.

complex128 checks the scheme. complex64 checks only the round-off floor:
soliton position, velocity and global phase are neutral modes, so float32
round-off accumulates in them instead of decaying. Over these 5000-10000 steps
the measured complex64 error is 1e-4 to 7e-3 depending on the problem and on
how often the norm is restored, hence the 1e-2 bound.
"""

import math

import pytest
import torch

import dipgpe
from util import C64, C128, DEVICE, max_abs


def bright_soliton(grid, g, eta, v, x0, t):
    """Bright soliton of i psi_t = -psi_xx / 2 + g |psi|^2 psi with g < 0."""
    x = grid.x64[0]
    psi = (eta / math.sqrt(-g) / torch.cosh(eta * (x - x0 - v * t))
           * torch.exp(1j * v * x + 0.5j * (eta ** 2 - v ** 2) * t))
    return psi.to(grid.dtype)


def gray_soliton_pair(grid, g, n0, v, d, t):
    """Two gray solitons at -d and +d moving towards each other with speed v.

    Each factor is the exact gray soliton i v/c + gamma tanh(gamma c (x - x_s(t)))
    with c = sqrt(g n0); opposite phase jumps keep the product periodic. The
    product is exact up to the soliton overlap ~exp(-2 gamma c separation).
    """
    x = grid.x64[0]
    c = math.sqrt(g * n0)
    beta, gamma = v / c, math.sqrt(1 - (v / c) ** 2)
    left = 1j * beta + gamma * torch.tanh(gamma * c * (x + d - v * t))
    right = -1j * beta + gamma * torch.tanh(gamma * c * (x - d + v * t))
    return (math.sqrt(n0) * left * right * complex(math.cos(g * n0 * t), -math.sin(g * n0 * t))).to(grid.dtype)


def evolve_in_chunks(stepper, psi, steps, chunk=10):
    """keep_norm restores the norm at the end of each evolve call."""
    for _ in range(steps // chunk):
        psi = stepper.evolve(psi, chunk)
    return stepper.evolve(psi, steps % chunk)


@pytest.mark.parametrize("dtype, tol", [(C64, 1e-2), (C128, 2e-6)])
def test_bright_soliton(dtype, tol):
    grid = dipgpe.Grid(512, 60.0, dtype=dtype, device=DEVICE)
    g, eta, v, x0, T, dt = -1.0, 1.0, 1.0, -5.0, 5.0, 1e-3
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    stepper = dipgpe.SplitStep(model, dt, keep_norm=dtype == C64)
    out = evolve_in_chunks(stepper, bright_soliton(grid, g, eta, v, x0, 0.0), round(T / dt))
    assert max_abs(out, bright_soliton(grid, g, eta, v, x0, T)) < tol


@pytest.mark.parametrize("dtype, tol", [(C64, 1e-2), (C128, 1e-6)])
def test_gray_soliton_pair(dtype, tol):
    grid = dipgpe.Grid(1024, 80.0, dtype=dtype, device=DEVICE)
    g, n0, v, d, T, dt = 1.0, 1.0, 0.5, 20.0, 10.0, 1e-3
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    stepper = dipgpe.SplitStep(model, dt, keep_norm=dtype == C64)
    out = evolve_in_chunks(stepper, gray_soliton_pair(grid, g, n0, v, d, 0.0), round(T / dt))
    assert max_abs(out, gray_soliton_pair(grid, g, n0, v, d, T)) < tol

"""Gate: free Gaussian spreading matches the analytic solution."""

import math

import pytest
import torch

import dipgpe
from util import C64, C128, DEVICE, rel_l2


def analytic_gaussian(grid, center, width, momentum, t):
    """Free evolution of dipgpe.gaussian: spreading plus Galilean drift (hbar = m = 1)."""
    a = 1 + 1j * t / (2 * width ** 2)
    psi = 1.0
    for x in grid.x64:
        psi = psi * ((2 * math.pi * width ** 2) ** -0.25 / a ** 0.5
                     * torch.exp(-(x - center - momentum * t) ** 2 / (4 * width ** 2 * a)
                                 + 1j * momentum * x - 0.5j * momentum ** 2 * t))
    return psi.to(grid.dtype)


@pytest.mark.parametrize("ndim, N, L", [(1, 256, 40.0), (2, 128, 30.0)])
@pytest.mark.parametrize("dtype, tol", [(C64, 1e-4), (C128, 1e-10)])
def test_free_gaussian(ndim, N, L, dtype, tol):
    grid = dipgpe.Grid((N,) * ndim, L, dtype=dtype, device=DEVICE)
    center, width, momentum, T, steps = -5.0, 1.0, 2.0, 2.0, 100
    psi = dipgpe.gaussian(grid, center, width, momentum)
    out = dipgpe.SplitStep(dipgpe.GPE(grid), T / steps).evolve(psi, steps)
    assert rel_l2(out, analytic_gaussian(grid, center, width, momentum, T)) < tol

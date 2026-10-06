"""Gate: the vortex detector finds known configurations exactly (count, sign, position).

Configurations are imprinted with dipgpe.fields (periodic phase + core ansatz),
whose zeros sit exactly at the prescribed points. Positions are compared with
the nearest periodic image.
"""

import pytest
import torch

import dipgpe
from dipgpe.fields import imprint_vortices, random_vortices, vortex_phase
from dipgpe.vortices import find_vortices, point_vortex_velocities
from util import C64, C128, DEVICE


def periodic_distance(a, b, lengths):
    d = torch.remainder(a[:, None, :] - b[None, :, :] + lengths / 2, lengths) - lengths / 2
    return d.pow(2).sum(-1).sqrt()


def imprinted(dtype, count, seed, shape=(256, 192), lengths=(64.0, 48.0)):
    grid = dipgpe.Grid(shape, lengths, dtype=dtype, device=DEVICE)
    positions, charges = random_vortices(grid, count, 6.0, seed=seed)
    psi = imprint_vortices(grid, torch.ones(grid.shape, dtype=dtype, device=DEVICE),
                           positions, charges, 1.0)
    return grid, psi, torch.tensor(positions, dtype=torch.float64), torch.tensor(charges)


@pytest.mark.parametrize("dtype", [C64, C128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detects_random_configurations(dtype, seed):
    grid, psi, expected, charges = imprinted(dtype, 12, seed)
    lengths = torch.tensor(grid.length, dtype=torch.float64)
    # Bilinear positions err by up to ~0.025 dx (core curvature, neighbours); refined
    # positions are limited by the spectral truncation of the core ansatz (~1e-6).
    for refine, tol in ((False, 0.05 * grid.dx[0]), (True, 1e-5)):
        found, q = find_vortices(grid, psi, refine=refine)
        assert len(found) == len(expected)
        d = periodic_distance(found.cpu(), expected, lengths)
        match = d.argmin(1)
        assert sorted(match.tolist()) == list(range(len(expected)))   # one-to-one
        assert torch.equal(q.cpu(), charges[match])                   # signs
        assert float(d.min(1).values.max()) < tol, (refine, float(d.min(1).values.max()))


def test_vortices_across_the_boundary_and_no_spurious_ones():
    grid = dipgpe.Grid((128, 128), 32.0, dtype=C128, device=DEVICE)
    positions = [(15.9, 0.3), (-15.7, 8.1), (0.2, 15.95), (5.1, -15.8)]
    charges = [1, -1, 1, -1]
    psi = imprint_vortices(grid, torch.ones(grid.shape, dtype=C128, device=DEVICE),
                           positions, charges, 1.0)
    found, q = find_vortices(grid, psi, refine=True)
    d = periodic_distance(found.cpu(), torch.tensor(positions, dtype=torch.float64),
                          torch.tensor(grid.length, dtype=torch.float64))
    assert len(found) == 4 and float(d.min(1).values.max()) < 1e-5
    assert torch.equal(q.cpu(), torch.tensor(charges)[d.argmin(1)])


def test_imprinted_field_is_smooth_and_periodic():
    grid, psi, _, _ = imprinted(C128, 10, 3)
    power = grid.fft(psi).abs() ** 2
    high = grid.k2.sqrt() > 0.7 * torch.pi / grid.dx[0]
    assert float(power[high].sum() / power.sum()) < 1e-12


def test_net_charge_must_vanish():
    grid = dipgpe.Grid((32, 32), 16.0, dtype=C128)
    with pytest.raises(ValueError):
        vortex_phase(grid, [(0.0, 0.0)], [1])


def test_density_threshold_ignores_empty_regions():
    grid = dipgpe.Grid((128, 128), 32.0, dtype=C128, device=DEVICE)
    rng = torch.Generator().manual_seed(0)
    noise = torch.randn(grid.shape, generator=rng, dtype=torch.complex128).to(DEVICE)
    cloud = torch.clamp(1 - grid.r2() / 100.0, min=0.0)      # exactly empty beyond r = 10
    psi = cloud * imprint_vortices(grid, torch.ones_like(noise), [(2.1, 0.3), (-2.9, 0.4)], [1, -1], 1.0)
    psi = psi + 1e-6 * noise
    assert len(find_vortices(grid, psi)[0]) > 2                       # noise outside the cloud
    found, q = find_vortices(grid, psi, min_density=1e-4)
    assert len(found) == 2 and int(q.sum()) == 0


def test_point_vortex_dipole_velocity_far_from_images():
    grid = dipgpe.Grid((64, 64), 10000.0, dtype=C128)   # images change v by ~(d/L)^2
    v = point_vortex_velocities(grid, [(0.0, 5.0), (0.0, -5.0)], [1, -1])
    assert torch.allclose(v, torch.tensor([[0.1, 0.0], [0.1, 0.0]], dtype=torch.float64), atol=1e-6)

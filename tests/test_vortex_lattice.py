"""Gate: the rotating ground state forms a vortex lattice consistent with the literature.

2D trap (omega = 1), g = 1000, rotating frame Omega, imaginary time from a
Gaussian with a random phase (so vortices can enter). Checks:
  * no vortices below the critical rotation (Omega = 0.2), count grows with Omega,
    all charges +1;
  * Abrikosov lattice with Feynman's density n_v = Omega / pi: the bulk
    nearest-neighbour spacing matches a = sqrt(2 pi / (sqrt(3) Omega));
  * the angular momentum approaches the rigid-body value N Omega <r^2> as the
    lattice fills the cloud.
Measured (complex64, N = 192): spacing 0.91 / 0.95 of Abrikosov and angular
momentum 0.86 / 0.93 of rigid at Omega = 0.6 / 0.8, both approaching 1.
Counting vortices inside a fixed disk is not used: with 10-20 vortices it
fluctuates by +-20% with the disk radius (measured), independent of convergence.
"""

import math

import torch

import dipgpe
from dipgpe.vortices import find_vortices
from util import C64, DEVICE

G = 1000.0


def lattice(omega, N=192, L=24.0):
    grid = dipgpe.Grid((N, N), L, dtype=C64, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(G), dipgpe.Rotation(omega))
    x, y = grid.x64
    rng = torch.Generator().manual_seed(0)
    noise = torch.randn(grid.shape, generator=rng, dtype=torch.float64).to(DEVICE)
    psi0 = dipgpe.normalize(grid, (torch.exp(-(x * x + y * y) / 32) * torch.exp(0.3j * noise)).to(C64))
    psi, _ = dipgpe.ground_state(model, psi0)
    positions, charges = find_vortices(grid, psi, min_density=1e-4 * float((psi.abs() ** 2).max()))
    positions, charges = positions.cpu(), charges.cpu()
    w2 = 1 - omega ** 2
    radius = math.sqrt(2 * math.sqrt(G * w2 / math.pi) / w2)          # Thomas-Fermi
    bulk = positions[positions.pow(2).sum(1).sqrt() < 0.6 * radius]
    d = torch.cdist(bulk, bulk) + 1e9 * torch.eye(len(bulk), dtype=torch.float64)
    spacing = float(d.min(1).values.mean()) if len(bulk) > 1 else float("nan")
    rigid = float(dipgpe.angular_momentum(grid, psi) / (omega * dipgpe.norm(grid, psi)
                                                     * dipgpe.expectation(grid, psi, grid.r2())))
    return charges, spacing, rigid


def test_vortex_lattice_matches_feynman_abrikosov_and_rigid_rotation():
    counts = {}
    for omega in (0.2, 0.6, 0.8):
        charges, spacing, rigid = lattice(omega)
        counts[omega] = len(charges)
        assert bool((charges == 1).all())
        if omega >= 0.6:
            a = math.sqrt(2 * math.pi / (math.sqrt(3) * omega))
            assert abs(spacing / a - 1) < 0.12, (omega, spacing / a)
            assert 0.8 < rigid < 1.0, (omega, rigid)
    assert counts[0.2] == 0 < counts[0.6] < counts[0.8], counts

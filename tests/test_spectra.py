"""Gate: the kinetic-energy decomposition sums to the total kinetic energy.

Also checks that spectra integrate to their energies and that the projection
separates known flows (an irrotational phase field is purely compressible).
"""

import pytest
import torch

import dipgpe
from dipgpe.fields import imprint_vortices, random_phase, random_vortices
from dipgpe.spectra import energy_spectra, kinetic_energies, velocity_fields
from util import C64, C128, DEVICE


def turbulent_field(dtype, shape=(256, 256), L=64.0):
    grid = dipgpe.Grid(shape, L, dtype=dtype, device=DEVICE)
    positions, charges = random_vortices(grid, 20, 4.0, seed=1)
    psi = imprint_vortices(grid, random_phase(grid, 1.0, 0.5, seed=2), positions, charges, 1.0)
    return grid, psi


@pytest.mark.parametrize("dtype, tol", [(C64, 2e-6), (C128, 1e-13)])
@pytest.mark.parametrize("shape, L", [((256, 256), 64.0), ((192, 128), (48.0, 40.0))])
def test_decomposition_sums_to_total(dtype, tol, shape, L):
    grid, psi = turbulent_field(dtype, shape, L)
    E = kinetic_energies(grid, psi)
    parts = E["incompressible"] + E["compressible"] + E["quantum"]
    assert abs(parts / E["total"] - 1) < tol
    assert abs(E["total"] / float(dipgpe.kinetic_energy(grid, psi)) - 1) < tol
    assert min(E["incompressible"], E["compressible"], E["quantum"]) > 0.01 * E["total"]


@pytest.mark.parametrize("dtype, tol", [(C64, 1e-6), (C128, 1e-13)])
def test_spectra_integrate_to_energies(dtype, tol):
    grid, psi = turbulent_field(dtype)
    E = kinetic_energies(grid, psi)
    k, spectra = energy_spectra(grid, psi)
    dk = float(k[1])
    for name, spectrum in spectra.items():
        assert abs(float(spectrum.sum()) * dk / E[name] - 1) < tol


def test_irrotational_flow_is_purely_compressible():
    grid = dipgpe.Grid((128, 128), 32.0, dtype=C128, device=DEVICE)
    psi = random_phase(grid, 1.0, 0.8, seed=4)          # uniform density, smooth phase
    E = kinetic_energies(grid, psi)
    assert E["incompressible"] < 1e-25 * E["compressible"]
    assert E["quantum"] < 1e-25 * E["compressible"]


def test_incompressible_part_is_divergence_free():
    grid, psi = turbulent_field(C128)
    w = velocity_fields(grid, psi)["incompressible"]
    w_k = grid.fft(w)
    div = sum(k * w_k[i] for i, k in enumerate(grid.k))
    # Nyquist rows are excluded from the projection by construction (see dipgpe.spectra).
    interior = sum((k.abs() < k.abs().max()).to(torch.float64) for k in grid.k) == grid.ndim
    assert float(div.abs()[interior].max()) < 1e-10 * float(w_k.abs().max())

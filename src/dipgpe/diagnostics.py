"""Integral diagnostics. Leading (batch, component) axes are kept in results."""

from __future__ import annotations

import torch


def density(psi):
    return psi.real * psi.real + psi.imag * psi.imag


def norm(grid, psi):
    """N = integral of |psi|^2."""
    return grid.integrate(density(psi))


def kinetic_energy(grid, psi):
    """1/2 integral |grad psi|^2, evaluated spectrally (Parseval)."""
    psi_k = grid.fft(psi)
    return 0.5 * (grid.k2 * density(psi_k)).sum(dim=grid.dims, dtype=torch.float64) * (grid.dV / grid.size)


def _term_energy(model, term, psi, rho, t):
    if hasattr(term, "energy_density"):
        return model.grid.integrate(term.energy_density(rho, t))
    return term.energy(model.grid, psi)


def energy_components(model, psi, t=0.0):
    """(kinetic, [energy of each term at time t]) in model.terms order."""
    rho = density(psi)
    kinetic = kinetic_energy(model.grid, psi)
    return kinetic, [_term_energy(model, term, psi, rho, t) for term in model.terms]


def energy(model, psi, t=0.0):
    """Total energy E[psi] at time t (not divided by N)."""
    kinetic, terms = energy_components(model, psi, t)
    return kinetic + sum(terms)


def chemical_potential(model, psi, t=0.0):
    """mu = <psi|H|psi> / N with H the (nonlinear) GPE operator at time t."""
    rho = density(psi)
    U = model.local_potential(rho, t)
    total = kinetic_energy(model.grid, psi) + model.grid.integrate(U * rho)
    for term in model.terms:
        if not hasattr(term, "local"):        # linear non-local terms (rotation)
            total = total + term.energy(model.grid, psi)
    return total / norm(model.grid, psi)


def residual(model, psi, t=0.0):
    """Eigenstate residual ||H psi - mu psi|| / ||psi|| with mu = <H>, at time t."""
    grid = model.grid
    h_psi = grid.ifft(0.5 * grid.k2 * grid.fft(psi)) + model.local_potential(density(psi), t) * psi
    for term in model.terms:
        if not hasattr(term, "local"):
            h_psi = h_psi + term.apply(grid, psi)
    mu = chemical_potential(model, psi, t)
    r = h_psi - mu.to(grid.real_dtype).reshape(mu.shape + (1,) * grid.ndim) * psi
    return torch.sqrt(norm(grid, r) / norm(grid, psi))


def expectation(grid, psi, f):
    """<f> = integral f |psi|^2 / N for a real field f."""
    rho = density(psi)
    return grid.integrate(f * rho) / grid.integrate(rho)


def angular_momentum(grid, psi):
    """<L_z> = integral psi* (-i)(x d_y - y d_x) psi (not divided by N)."""
    psi_k = grid.fft(psi)
    d_x = grid.ifft(1j * grid.k[0] * psi_k)
    d_y = grid.ifft(1j * grid.k[1] * psi_k)
    return grid.integrate((psi.conj() * (-1j) * (grid.x[0] * d_y - grid.x[1] * d_x)).real)


def center_of_mass(grid, psi):
    """<x_i> for each axis, stacked along the last dimension."""
    return torch.stack([expectation(grid, psi, xi) for xi in grid.x], dim=-1)

"""The ground-state minimizer: exact gradient, kept norm, batches."""

import math

import pytest
import torch

import dipgpe
from dipgpe.minimize import energy_and_gradient
from util import C64, C128, DEVICE


def directional_check(model, psi, eps=1e-6):
    """dE/ds along a random direction q equals 2 Re <q, H psi> (central difference)."""
    grid = model.grid
    gen = torch.Generator().manual_seed(1)
    q = torch.complex(torch.randn(grid.shape, generator=gen, dtype=torch.float64),
                      torch.randn(grid.shape, generator=gen, dtype=torch.float64)).to(grid.device, grid.dtype)
    q = q * psi.abs()                                       # stay where the state lives
    _, h = energy_and_gradient(model, psi)
    E_p, _ = energy_and_gradient(model, psi + eps * q, gradient=False)
    E_m, _ = energy_and_gradient(model, psi - eps * q, gradient=False)
    numeric = float(E_p - E_m) / (2 * eps)
    exact = 2 * float(grid.integrate((q.conj() * h).real))
    return numeric, exact


def test_gradient_matches_energy_2d_rotation_lhy():
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(50.0), dipgpe.LHY(3.0), dipgpe.Rotation(0.4))
    x, y = grid.x64
    psi = dipgpe.normalize(grid, (dipgpe.gaussian(grid, 0.0, 1.5) * torch.exp(0.5j * torch.atan2(y, x + 0.3))).to(C128))
    numeric, exact = directional_check(model, psi)
    assert abs(numeric - exact) < 1e-7 * abs(exact)
    E, _ = energy_and_gradient(model, psi)
    assert abs(float(E - dipgpe.energy(model, psi))) < 1e-12 * abs(float(E))


def test_gradient_matches_energy_3d_dipolar():
    grid = dipgpe.Grid((24, 24, 24), 12.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0),
                   dipgpe.Dipolar(grid, 15.0, direction=(0, 1, 0), cutoff="tube"))
    psi = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.4).to(C128), 10.0)
    numeric, exact = directional_check(model, psi)
    assert abs(numeric - exact) < 1e-7 * abs(exact)
    E, _ = energy_and_gradient(model, psi)
    assert abs(float(E - dipgpe.energy(model, psi))) < 1e-12 * abs(float(E))


@pytest.mark.parametrize("dtype, res_tol, norm_tol", [(C64, 1e-4, 1e-6), (C128, 1e-10, 1e-13)])
def test_norm_kept_and_batch_converges_per_entry(dtype, res_tol, norm_tol):
    grid = dipgpe.Grid((64, 64), 16.0, dtype=dtype, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(100.0))
    seeds = torch.stack([dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.0), 2.0),
                         dipgpe.normalize(grid, dipgpe.translate(grid, dipgpe.gaussian(grid, 0.0, 2.5), (1.0, -0.5)), 2.0)])
    psi, info = dipgpe.ground_state(model, seeds, tol=res_tol)
    assert info["residual"].shape == (2,)
    assert float(dipgpe.residual(model, psi).max()) <= res_tol * 1.01
    assert float((dipgpe.norm(grid, psi) / 2.0 - 1).abs().max()) < norm_tol
    E = dipgpe.energy(model, psi)
    assert abs(float(E[0] - E[1])) < (1e-6 if dtype == C64 else 1e-12) * abs(float(E[0]))


def test_converges_in_few_iterations_independent_of_resolution():
    iterations = []
    for n in (64, 256):
        grid = dipgpe.Grid((n, n), 16.0, dtype=C128, device=DEVICE)
        model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(200.0))
        _, info = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5)))
        iterations.append(info["iterations"])
    assert max(iterations) < 150 and iterations[1] < 1.5 * iterations[0], iterations


def test_twist_gradient_and_superfluid_fraction_of_uniform_gas():
    grid = dipgpe.Grid((32, 48), (12.0, 10.0), dtype=C128, device=DEVICE)
    x, _ = grid.x64
    model = dipgpe.GPE(grid, dipgpe.Potential((0.5 * x * x).expand(grid.shape).to(grid.real_dtype)), dipgpe.Contact(30.0))
    psi0 = dipgpe.normalize(grid, torch.exp(-x * x / 2).expand(grid.shape).to(C128), 5.0)
    E0 = float(dipgpe.energy(model, dipgpe.ground_state(model, psi0)[0]))
    q = 0.3
    twisted = dipgpe.GPE(grid, *model.terms, dipgpe.Twist(q))
    psi, _ = dipgpe.ground_state(twisted, psi0)
    numeric, exact = directional_check(twisted, psi * (1 + 0.1 * torch.cos(grid.x64[1])).to(C128))
    assert abs(numeric - exact) < 1e-7 * abs(exact)
    f_s = (float(dipgpe.energy(twisted, psi)) - E0) / (0.5 * q * q * 5.0)    # Galilean invariance: f_s = 1
    assert abs(f_s - 1) < 1e-9
    with pytest.raises(NotImplementedError):
        dipgpe.SplitStep(twisted, 0.01)


def test_non_strict_returns_unconverged_state():
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(100.0))
    psi0 = dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5))
    with pytest.raises(RuntimeError):
        dipgpe.ground_state(model, psi0, max_iter=5)
    psi, info = dipgpe.ground_state(model, psi0, max_iter=5, strict=False)
    assert not info["converged"] and info["iterations"] == 5 and psi.shape == psi0.shape


def test_resample_is_exact_for_band_limited_fields():
    coarse = dipgpe.Grid((32, 24), (12.0, 10.0), dtype=C64, device=DEVICE)
    fine = dipgpe.Grid((64, 60), (12.0, 10.0), dtype=C128, device=DEVICE)
    def field(g):
        x, y = g.x64
        return (torch.exp(-(x * x) / 2 - (y * y) / 1.5) * torch.exp(0.7j * x)).to(g.dtype)
    up = dipgpe.resample(field(coarse), coarse, fine)
    assert up.dtype == C128 and float((up - field(fine)).abs().max()) < 2e-6    # complex64 source
    back = dipgpe.resample(field(fine), fine, coarse)
    assert float((back - field(coarse)).abs().max()) < 1e-6
    same = dipgpe.Grid((64, 60), (12.0, 10.0), dtype=C128, device=DEVICE)
    assert float((dipgpe.resample(field(fine), fine, same) - field(fine)).abs().max()) < 1e-12


def test_combined_preconditioner_same_state_fewer_iterations():
    """A single central vortex (unique minimum; from random seeds the two
    preconditioners can end in different metastable lattices)."""
    grid = dipgpe.Grid((128, 128), 20.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(400.0), dipgpe.Rotation(0.5))
    x, y = grid.x64
    seed = dipgpe.normalize(grid, (torch.exp(-(x * x + y * y) / 18) * torch.exp(1j * torch.atan2(y, x))).to(C128))
    out = {pc: dipgpe.ground_state(model, seed, tol=1e-8, preconditioner=pc) for pc in ("kinetic", "combined")}
    E = {pc: float(dipgpe.energy(model, psi)) for pc, (psi, _) in out.items()}
    assert abs(E["kinetic"] - E["combined"]) < 1e-10 * abs(E["kinetic"])
    assert out["combined"][1]["iterations"] < out["kinetic"][1]["iterations"]


def test_geodesic_reuse_matches_full_evaluation():
    """E and H psi along cos(t) psi + sin(t) q from cached operator actions equal a
    full evaluation (dipolar + LHY + rotation + potential), and the minimizer with
    and without reuse ends in the same state."""
    from dipgpe.minimize import _Geodesic
    grid = dipgpe.Grid((24, 24, 16), (12.0, 12.0, 8.0), dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0), dipgpe.LHY(2.0),
                   dipgpe.Dipolar(grid, 15.0, direction=(0, 0, 1), cutoff="cylinder"), dipgpe.Rotation(0.3))
    x, y, _ = grid.x64
    psi = dipgpe.normalize(grid, (dipgpe.gaussian(grid, 0.0, 1.4) * torch.exp(1j * torch.atan2(y, x + 0.2))).to(C128), 10.0)
    gen = torch.Generator().manual_seed(3)
    q = torch.complex(torch.randn(grid.shape, generator=gen, dtype=torch.float64),
                      torch.randn(grid.shape, generator=gen, dtype=torch.float64)).to(grid.device, C128) * psi.abs()
    geo = _Geodesic(model, 0.0)
    state, dirn = geo.state(psi), geo.direction(psi, q)
    for theta in (0.03, 0.4):
        c, s = math.cos(theta), math.sin(theta)
        f = c * psi + s * q
        E_geo, h_geo = geo.evaluate(f, geo.along(state, dirn, c, s))
        E, h = energy_and_gradient(model, f)
        assert abs(float(E_geo - E)) < 1e-12 * abs(float(E))
        assert float((h_geo - h).abs().max()) < 1e-10 * float(h.abs().max())
    out = {r: dipgpe.ground_state(model, psi, tol=1e-9, reuse=r, preconditioner="combined") for r in (True, False)}
    E = {r: float(dipgpe.energy(model, p)) for r, (p, _) in out.items()}
    assert abs(E[True] - E[False]) < 1e-11 * abs(E[False])
    assert 0.0 <= float(out[True][1]["capped"]) <= 1.0

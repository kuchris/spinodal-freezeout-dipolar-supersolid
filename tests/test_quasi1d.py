"""Quasi-1D tube model: E1, the reduced dipolar kernel, and the roton softening
point of Kirkby et al. (PRR 7, 2025): a_c ~ 91.05 a0 at n = 2500 /um with
l = 1.08 um, eta = 4.25 (164Dy, omega_perp = 2 pi x 150 Hz). Their closed-form
kernel reproduces it; the exact kernel (direct transverse integral) differs for
eta != 1 and gives a_c = 90.54 a0."""

import math

import torch

import dipgpe
from dipgpe.quasi1d import dipolar_kernel_1d, dipolar_kernel_1d_closed, exp_e1, transverse_energy, tube_model
from dipgpe.terms import lhy_coefficient
from dipgpe.units import OscillatorUnits
from util import C128, DEVICE


def test_exp_e1_against_reference_values():
    ref = {0.1: 1.8229239584193906, 1.0: 0.21938393439552029, 5.0: 0.0011482955912753257,
           10.0: 4.156968929685324e-06, 30.0: 3.0215520106888125e-15}
    x = torch.tensor(list(ref), dtype=torch.float64)
    got = exp_e1(x) * torch.exp(-x)
    for g, (xv, e1) in zip(got.tolist(), ref.items()):
        assert abs(g / e1 - 1) < 1e-12, (xv, g, e1)


def direct_kernel(kx, g_dd, ell, eta, n=2401):
    sy, sz = ell / math.sqrt(eta), ell * math.sqrt(eta)
    ky = torch.linspace(-14 / sy, 14 / sy, n, dtype=torch.float64)
    kz = torch.linspace(-14 / sz, 14 / sz, n, dtype=torch.float64)
    KY, KZ = torch.meshgrid(ky, kz, indexing="ij")
    w = (ky[1] - ky[0]) * (kz[1] - kz[0]) / (2 * math.pi) ** 2
    return float((g_dd * (3 * KZ ** 2 / (kx ** 2 + KY ** 2 + KZ ** 2) - 1)
                  * torch.exp(-(KY ** 2 * sy ** 2 + KZ ** 2 * sz ** 2) / 2)).sum() * w)


def test_kernel_matches_direct_transverse_integral():
    g_dd, ell = 0.9, 1.3
    for eta in (1.0, 4.25):
        for kx in (0.3, 1.0, 3.0):
            exact = float(dipolar_kernel_1d(torch.tensor([kx], dtype=torch.float64), g_dd, ell, eta))
            assert abs(exact - direct_kernel(kx, g_dd, ell, eta)) < 1e-12, (eta, kx)
        zero = float(dipolar_kernel_1d(torch.tensor([0.0], dtype=torch.float64), g_dd, ell, eta))
        assert abs(zero - g_dd * (2 - eta) / (2 * math.pi * ell ** 2 * (1 + eta))) < 1e-13
    far = float(dipolar_kernel_1d(torch.tensor([1e4], dtype=torch.float64), g_dd, ell, 4.25))
    assert abs(far + g_dd / (2 * math.pi * ell ** 2)) < 1e-6 * g_dd


def test_closed_form_is_exact_only_for_a_round_cross_section():
    g_dd, ell = 0.9, 1.3
    k = torch.tensor([0.0, 0.3, 1.0, 3.0], dtype=torch.float64)
    assert float((dipolar_kernel_1d_closed(k, g_dd, 1.0 * ell, 1.0) - dipolar_kernel_1d(k, g_dd, ell, 1.0)).abs().max()) < 1e-13
    diff = (dipolar_kernel_1d_closed(k, g_dd, ell, 4.25) - dipolar_kernel_1d(k, g_dd, ell, 4.25)).abs()
    assert float(diff[0]) < 1e-13 and float(diff[1:].max()) > 1e-3 * g_dd       # equal at k = 0 only


def roton_gap_squared(a_s_bohr, units, kernel, n_um=2500.0, ell_um=1.08, eta=4.25):
    """min_k of the uniform-state Bogoliubov omega^2 of the reduced model."""
    a_s, a_dd = units.bohr(a_s_bohr), units.bohr(130.8)
    ell = ell_um / units.microns(1.0)
    n = units.per_micron(n_um)
    g1 = 4 * math.pi * a_s / (2 * math.pi * ell ** 2)
    gam1 = 2 * lhy_coefficient(a_s, a_dd) / (5 * math.pi ** 1.5 * ell ** 3)
    k = torch.linspace(1e-3, 6.0, 6000, dtype=torch.float64)
    e = 0.5 * k * k
    w2 = e * (e + 2 * n * (g1 + kernel(k, 4 * math.pi * a_dd, ell, eta) + 1.5 * gam1 * math.sqrt(n)))
    return float(w2.min())


def critical_scattering_length(kernel):
    units = OscillatorUnits(164, 2 * math.pi * 150)
    lo, hi = 88.0, 96.0
    assert roton_gap_squared(lo, units, kernel) < 0 < roton_gap_squared(hi, units, kernel)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if roton_gap_squared(mid, units, kernel) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_reduced_model_critical_scattering_length():
    assert abs(critical_scattering_length(dipolar_kernel_1d_closed) - 91.05) < 0.05    # Kirkby et al.
    assert abs(critical_scattering_length(dipolar_kernel_1d) - 90.544) < 0.002         # exact kernel


def test_reduced_model_ground_state_and_energy_offset():
    units = OscillatorUnits(164, 2 * math.pi * 150)
    ell, eta = 1.08 / units.microns(1.0), 4.25
    L = 2.69 / units.microns(1.0)
    grid = dipgpe.Grid(64, L, dtype=C128, device=DEVICE)
    model = tube_model(grid, units.bohr(89.0), units.bohr(130.8), ell, eta)
    x = grid.x64[0]
    N = units.per_micron(2500.0) * L
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, (1 + 0.3 * torch.cos(2 * math.pi * x / L)).to(C128), N))
    line = psi.abs() ** 2
    assert float((line.max() - line.min()) / (line.max() + line.min())) > 0.1   # modulated below a_c
    assert transverse_energy(1.0, 1.0) == 1.0                                    # isotropic oscillator

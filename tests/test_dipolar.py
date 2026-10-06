"""Gate: the dipolar energy of a known density matches an independent calculation.

For a Gaussian density n = N exp(-rho^2/s_r^2 - z^2/s_z^2) / (pi^1.5 s_r^2 s_z), with
dipoles along z,
    E_dd = 1/2 int d^3k / (2 pi)^3 U_dd(k) |n(k)|^2
         = (N^2 g_dd / 2) (2 pi / (2 pi)^3) sqrt(pi / 2) int_{-1}^{1} (3 mu^2 - 1) a(mu)^(-3/2) d mu,
    a(mu) = s_r^2 (1 - mu^2) + s_z^2 mu^2,
after the radial k integral is done analytically; the mu integral uses
Gauss-Legendre quadrature, independent of the FFT grid. Also checks the LHY
coefficient and that every term's potential is the density derivative of its
energy.
"""

import math

import pytest
import torch

import dipgpe
from dipgpe.terms import _gauss_legendre
from util import C128, DEVICE


def gaussian_density(grid, s_r, s_z, N=1.0):
    x, y, z = grid.x64
    n = N * torch.exp(-(x * x + y * y) / s_r ** 2 - z * z / s_z ** 2) / (math.pi ** 1.5 * s_r ** 2 * s_z)
    return n.to(grid.real_dtype)


def gaussian_dipolar_energy(s_r, s_z, g_dd, N=1.0):
    mu, w = _gauss_legendre(400)
    a = s_r ** 2 * (1 - mu ** 2) + s_z ** 2 * mu ** 2
    integral = float((w * (3 * mu ** 2 - 1) * a ** -1.5).sum())
    return N ** 2 * g_dd / 2 * (2 * math.pi / (2 * math.pi) ** 3) * math.sqrt(math.pi / 2) * integral


# Widths keep |n(k)|^2 below 1e-12 at the grid Nyquist (s = 0.6 left 1e-7 and a 1e-6 error).
@pytest.mark.parametrize("s_r, s_z", [(1.0, 1.8), (1.5, 0.8), (1.2, 1.2)])
def test_gaussian_dipolar_energy(s_r, s_z):
    grid = dipgpe.Grid((96, 96, 96), 32.0, dtype=C128, device=DEVICE)
    g_dd = 0.7
    term = dipgpe.Dipolar(grid, g_dd)                      # spherical cutoff R = 16
    n = gaussian_density(grid, s_r, s_z)
    E = float(grid.integrate(term.energy_density(n)))
    exact = gaussian_dipolar_energy(s_r, s_z, g_dd)
    if s_r == s_z:
        assert abs(E) < 1e-12 and abs(exact) < 1e-12   # isotropic: dipolar energy vanishes
    else:
        assert abs(E / exact - 1) < 1e-8, (E, exact)
    assert (E < 0) == (s_z > s_r) or s_r == s_z        # elongated along the dipoles: attractive


def test_dipole_direction_is_a_rotation():
    grid = dipgpe.Grid((64, 64, 64), 32.0, dtype=C128, device=DEVICE)
    along_z = gaussian_density(grid, 1.0, 1.6)
    along_x = along_z.permute(2, 1, 0).contiguous()          # same cloud elongated along x
    E_z = grid.integrate(dipgpe.Dipolar(grid, 1.0).energy_density(along_z))
    E_x = grid.integrate(dipgpe.Dipolar(grid, 1.0, direction=(1, 0, 0)).energy_density(along_x))
    assert abs(float(E_x / E_z) - 1) < 1e-10


def test_lhy_coefficient():
    a_s, a_dd = 0.01, 0.013
    expected = 32 / 3 * 4 * math.pi * a_s * math.sqrt(a_s ** 3 / math.pi) * dipgpe.q5(a_dd / a_s)
    assert abs(dipgpe.lhy_coefficient(a_s, a_dd) / expected - 1) < 1e-14
    assert abs(dipgpe.q5(0.0) - 1) < 1e-12 and abs(dipgpe.q5(1.0) - 3 ** 2.5 / 6) < 1e-10


def test_potentials_are_energy_derivatives():
    """d/dl E[(1 + l) psi] at l = 0 equals 2 mu N for every term combination."""
    grid = dipgpe.Grid((48, 48, 48), 16.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid, (1.0, 1.0, 0.5))), dipgpe.Contact(0.3),
                   dipgpe.Dipolar(grid, 0.4, direction=(0, 1, 0)), dipgpe.LHY(0.05))
    psi = dipgpe.gaussian(grid, (0.3, -0.2, 0.1), 1.1, (0.2, 0.0, 0.3)) * 30
    h = 1e-5
    dE = (float(dipgpe.energy(model, psi * (1 + h))) - float(dipgpe.energy(model, psi * (1 - h)))) / (2 * h)
    two_mu_N = 2 * float(dipgpe.chemical_potential(model, psi) * dipgpe.norm(grid, psi))
    assert abs(dE / two_mu_N - 1) < 1e-8


def isolated_tube_energy(lam, eps, k0, s_x, s_y, alpha, g_dd):
    """Energy per length of n = lam (1 + eps cos k0 z) exp(-x^2/s_x^2 - y^2/s_y^2) / (pi s_x s_y)
    in infinite space (no images), dipoles along (cos alpha, sin alpha, 0):
        e = lam^2 / 2 int d^2k / (2 pi)^2 |G(k)|^2 [U(k, 0) + eps^2 / 2 U(k, k0)],
    |G|^2 = exp(-k^2 s(phi) / 2), s = s_x^2 cos^2 phi + s_y^2 sin^2 phi, by polar quadrature."""
    phi, wphi = _gauss_legendre(256)
    phi, wphi = math.pi * (phi + 1), math.pi * wphi                 # [0, 2 pi]
    c2 = torch.cos(phi - alpha) ** 2
    s = s_x ** 2 * torch.cos(phi) ** 2 + s_y ** 2 * torch.sin(phi) ** 2
    uniform = float((wphi * (3 * c2 - 1) / s).sum())                # radial integral = 1 / s
    t, wt = _gauss_legendre(400)
    kmax = 12.0 / math.sqrt(min(s_x, s_y) ** 2)
    k, wk = kmax * (t + 1) / 2, kmax * wt / 2                       # radial nodes on [0, kmax]
    K, S, C2 = k[:, None], s[None, :], c2[None, :]
    integrand = K * torch.exp(-K ** 2 * S / 2) * (3 * K ** 2 * C2 / (K ** 2 + k0 ** 2) - 1)
    modulated = float((wk[:, None] * wphi[None, :] * integrand).sum())
    return lam ** 2 / 2 * g_dd / (2 * math.pi) ** 2 * (uniform + eps ** 2 / 2 * modulated)


@pytest.mark.parametrize("alpha", [math.pi / 2, 0.3])
def test_tube_cutoff_matches_isolated_tube(alpha):
    """A periodic-in-z tube with transverse cutoff equals an isolated infinite tube."""
    Lp, Lz, s_x, s_y, eps, g_dd, lam = 24.0, 6.0, 0.8, 1.5, 0.5, 0.7, 1.0
    grid = dipgpe.Grid((96, 96, 32), (Lp, Lp, Lz), dtype=C128, device=DEVICE)
    x, y, z = grid.x64
    k0 = 2 * math.pi / Lz
    n = (lam * (1 + eps * torch.cos(k0 * z)) * torch.exp(-x * x / s_x ** 2 - y * y / s_y ** 2)
         / (math.pi * s_x * s_y)).to(torch.float64)
    term = dipgpe.Dipolar(grid, g_dd, direction=(math.cos(alpha), math.sin(alpha), 0.0), cutoff="tube")
    e_grid = float(grid.integrate(term.energy_density(n))) / Lz
    e_exact = isolated_tube_energy(lam, eps, k0, s_x, s_y, alpha, g_dd)
    assert abs(e_grid / e_exact - 1) < 1e-7, (e_grid, e_exact)
    # Without the cutoff, transverse images change the energy at the percent level.
    e_periodic = float(grid.integrate(dipgpe.Dipolar(grid, g_dd, direction=(math.cos(alpha), math.sin(alpha), 0.0),
                                                  cutoff=None).energy_density(n))) / Lz
    assert abs(e_periodic / e_exact - 1) > 1e-3


def test_cylinder_cutoff_matches_sphere_cutoff_for_a_flat_cloud():
    """Same cloud, cubic box with the sphere cutoff vs a box half as tall with the
    cylindrical cutoff (dipoles along z): equal dipolar energies, continuous k -> 0."""
    def energy(grid, cutoff):
        x, y, z = grid.x64
        rho = (100.0 * torch.exp(-(x * x + y * y) / (2 * 1.2 ** 2) - z * z / (2 * 0.8 ** 2))).to(grid.real_dtype)
        d = dipgpe.Dipolar(grid, 0.3, direction=(0, 0, 1), cutoff=cutoff)
        return float(grid.integrate(0.5 * d.potential(rho) * rho)), d
    cube = dipgpe.Grid((64, 64, 64), 16.0, dtype=C128, device=DEVICE)
    flat = dipgpe.Grid((64, 64, 40), (16.0, 16.0, 10.0), dtype=C128, device=DEVICE)
    e_sphere, _ = energy(cube, "sphere")
    e_cyl, d = energy(flat, "cylinder")
    assert abs(e_cyl / e_sphere - 1) < 1e-6, (e_cyl, e_sphere)
    zero = 0.3 * (2 - 3 * d.height / math.sqrt(d.radius ** 2 + d.height ** 2))
    for kx, kz in ((1e-4, 0.0), (0.0, 1e-4), (1e-4, 1e-4)):
        k = [torch.full((1, 1, 1), kx, dtype=torch.float64, device=DEVICE), torch.zeros(1, 1, 1, dtype=torch.float64,
             device=DEVICE), torch.full((1, 1, 1), kz, dtype=torch.float64, device=DEVICE)]
        assert abs(float(d._kernel(k)) - zero) < 1e-5


def test_cylinder_kernel_matches_direct_integral():
    """Kernel values at finite k against a direct integral over the cylinder,
    independent of the slab/annulus split (suggested by an external review):
    K(k) = K(0) + 3 g int_0^1 (1 - 3t^2) int_0^s(t) [J0(k_rho r sqrt(1-t^2)) cos(k_z r t) - 1] dr/r dt,
    s(t) = min(R / sqrt(1-t^2), Z / t), t = cos(theta)."""
    from dipgpe.terms import _gauss_legendre
    g, R, Z = 0.7, 8.0, 5.0
    grid = dipgpe.Grid((32, 32, 16), (16.0, 16.0, 10.0), dtype=C128, device=DEVICE)
    d = dipgpe.Dipolar(grid, g, direction=(0, 0, 1), cutoff="cylinder", radius=R, height=Z)
    zero = g * (2 - 3 * Z / math.sqrt(R * R + Z * Z))
    u0, w0 = _gauss_legendre(16)

    def composite(a, b, panels):
        edges = torch.linspace(a, b, panels + 1, dtype=torch.float64)
        lo, h = edges[:-1, None], (edges[1:] - edges[:-1])[:, None]
        return (lo + h * (u0 + 1) / 2).reshape(-1), (h / 2 * w0).reshape(-1)

    t_star = Z / math.sqrt(R * R + Z * Z)
    t1, wt1 = composite(0.0, t_star, 40)
    t2, wt2 = composite(t_star, 1.0, 40)
    t, wt = torch.cat([t1, t2]), torch.cat([wt1, wt2])
    s = torch.minimum(R / torch.sqrt(1 - t * t), Z / t)
    u, wu = composite(0.0, 1.0, 40)                          # r = s(t) u, dr / r = du / u
    r = s[:, None] * u[None, :]
    for kr, kz in ((0.3, 0.0), (0.0, 0.4), (1.0, 0.7), (2.0, 1.5), (0.05, 0.02)):
        f = (torch.special.bessel_j0(kr * r * torch.sqrt(1 - t * t)[:, None]) * torch.cos(kz * r * t[:, None]) - 1) / u
        direct = zero + 3 * g * float((wt * (1 - 3 * t * t) * (f @ wu)).sum())
        k = [torch.full((1, 1, 1), kr, dtype=torch.float64, device=DEVICE),
             torch.zeros(1, 1, 1, dtype=torch.float64, device=DEVICE),
             torch.full((1, 1, 1), kz, dtype=torch.float64, device=DEVICE)]
        assert abs(float(d._kernel(k)) - direct) < 1e-8, (kr, kz, float(d._kernel(k)), direct)

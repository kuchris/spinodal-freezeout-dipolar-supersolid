"""Quasi-1D (reduced) model of a dipolar gas in a tube.

The transverse wavefunction is a fixed Gaussian (Blakie et al., arXiv:2004.12577;
Kirkby et al., PRR 7 (2025), arXiv:2411.18395):

    phi(y, z) = exp[-(eta y^2 + z^2 / eta) / (2 l^2)] / (l sqrt(pi)),

tube axis = the 1D grid axis, dipoles along z (transverse), so the amplitude
widths are s_y = l / sqrt(eta) and s_z = l sqrt(eta). Integrating it out gives
1D contact g / (2 pi l^2), LHY 2 gamma / (5 pi^(3/2) l^3) |psi|^3 and a 1D
dipolar kernel, computed exactly by quadrature (dipolar_kernel_1d). Kirkby et
al. Eq. (3) give the closed form (with mu0 mu_m^2 = 3 g_dd)

    U(k) = g_dd [4 - 2 eta - 3 sqrt(eta) k^2 l^2 e^u E1(u)] / (4 pi l^2 (1 + eta)),
    u = sqrt(eta) k^2 l^2 / 2

(their Eq. (3) as printed reads Ei(-u); the sign that gives the right limits,
U(0) = g_dd (2 - eta) / (2 pi l^2 (1 + eta)) and U(k -> inf) = -g_dd / (2 pi l^2),
is E1(u) = -Ei(-u) in this form). It is exact for eta = 1 but not for an
elliptical cross-section (dipolar_kernel_1d_closed; tests/test_quasi1d.py).

The transverse kinetic and trap energy per atom is a constant
(transverse_energy), needed only to compare energies with 3D or between (l, eta).
"""

from __future__ import annotations

import math

import torch

from .model import GPE
from .terms import LHY, Contact, Potential, _gauss_legendre, lhy_coefficient

EULER = 0.5772156649015329


def exp_e1(x):
    """e^x E1(x) for x >= 0 (float64 tensor; E1(x) = -Ei(-x)); 0 at x = 0 is
    replaced by +inf. Series for x <= 1, continued fraction (Lentz) above."""
    x = torch.as_tensor(x, dtype=torch.float64)
    out = torch.empty_like(x)
    small = x <= 1.0
    xs = x[small]
    # E1(x) = -gamma - ln x - sum_{n>=1} (-x)^n / (n n!)
    term = torch.ones_like(xs)
    s = torch.zeros_like(xs)
    for n in range(1, 40):
        term = term * (-xs) / n
        s = s + term / n
    out[small] = torch.exp(xs) * (-EULER - torch.log(xs) - s)
    xl = x[~small]
    # e^x E1(x) = 1 / (x + 1 - 1 / (x + 3 - 4 / (x + 5 - ...)))  (modified Lentz)
    b = xl + 1.0
    c = torch.full_like(xl, 1e300)
    d = 1.0 / b
    h = d.clone()
    for i in range(1, 200):
        a = -float(i * i)
        b = b + 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        delta = c * d
        h = h * delta
    out[~small] = h
    return out


def dipolar_kernel_1d(k, g_dd, ell, eta, nodes=400):
    """Exact U(k) of the quasi-1D dipolar interaction (dipoles along z), float64:
    U = g_dd [3 I(k) - 1 / (2 pi s_y s_z)] with
    I(k) = int_0^inf exp(-t k^2) / (8 pi sqrt(s_y^2/2 + t) (s_z^2/2 + t)^(3/2)) dt
    (from 1/q^2 = int_0^inf exp(-t q^2) dt; Gauss-Legendre in t = c s / (1 - s))."""
    k = torch.as_tensor(k, dtype=torch.float64)
    sy2, sz2 = ell * ell / eta, ell * ell * eta
    u, w = _gauss_legendre(nodes)
    s = (u + 1) / 2
    c = 0.5 * math.sqrt(sy2 * sz2)
    t = c * s / (1 - s)
    dt = c / (1 - s) ** 2 * (w / 2)
    f = dt / (8 * math.pi * torch.sqrt(sy2 / 2 + t) * (sz2 / 2 + t) ** 1.5)
    I = (torch.exp(-k.reshape(-1, 1) ** 2 * t) * f).sum(-1).reshape(k.shape)
    return g_dd * (3 * I - 1 / (2 * math.pi * math.sqrt(sy2 * sz2)))


def dipolar_kernel_1d_closed(k, g_dd, ell, eta):
    """Closed form of Kirkby et al. Eq. (3) (with E1, see the module docstring).
    Exact for eta = 1 and in the limits k -> 0, inf; for eta = 4.25 it deviates
    from the exact kernel by up to ~2% at intermediate k (tests/test_quasi1d.py)."""
    k = torch.as_tensor(k, dtype=torch.float64)
    u = math.sqrt(eta) * k * k * ell * ell / 2
    tail = torch.where(u > 0, 2 * u * exp_e1(torch.where(u > 0, u, torch.ones_like(u))),
                       torch.zeros_like(u))          # x e^x E1(x) -> 0 as x -> 0
    return g_dd * (4 - 2 * eta - 3 * tail) / (4 * math.pi * ell * ell * (1 + eta))


class TubeDipolar1D:
    """Quasi-1D dipolar interaction on a 1D grid (periodic along the tube).
    ``kernel="exact"`` (default) or ``"closed"`` (Kirkby et al. Eq. (3), to
    reproduce their results)."""

    nonlinear = True
    linear_in_density = True
    energy_fraction = 0.5

    def __init__(self, grid, g_dd, ell, eta, kernel="exact"):
        if grid.ndim != 1:
            raise ValueError("TubeDipolar1D needs a 1D grid")
        if kernel not in ("exact", "closed"):
            raise ValueError("kernel must be 'exact' or 'closed'")
        self.grid, self.g_dd, self.ell, self.eta = grid, float(g_dd), float(ell), float(eta)
        k = 2 * math.pi * torch.fft.rfftfreq(grid.shape[0], d=grid.dx[0], dtype=torch.float64)
        f = dipolar_kernel_1d if kernel == "exact" else dipolar_kernel_1d_closed
        self.kernel = f(k, g_dd, ell, eta).to(grid.device, grid.real_dtype)

    def potential(self, density):
        return torch.fft.irfft(torch.fft.rfft(density, dim=-1) * self.kernel, n=self.grid.shape[0], dim=-1)

    def local(self, density, t=0.0):
        return self.potential(density)

    def energy_density(self, density, t=0.0):
        return 0.5 * self.potential(density) * density


def transverse_energy(ell, eta, omega_y=1.0, omega_z=1.0):
    """Transverse kinetic + trap energy per atom of the Gaussian (s_y = l / sqrt(eta),
    s_z = l sqrt(eta)): (1/s_y^2 + 1/s_z^2) / 4 + (omega_y^2 s_y^2 + omega_z^2 s_z^2) / 4."""
    sy2, sz2 = ell * ell / eta, ell * ell * eta
    return 0.25 * (1 / sy2 + 1 / sz2) + 0.25 * (omega_y ** 2 * sy2 + omega_z ** 2 * sz2)


def tube_model(grid, a_s, a_dd, ell, eta, potential=None, kernel="exact"):
    """GPE on a 1D grid for the reduced tube model. ``a_s`` (in units of l) may be
    a callable t -> a_s(t) for ramps; ``a_dd`` is fixed. Energies exclude
    transverse_energy."""
    if callable(a_s):
        g = lambda t: 4 * math.pi * a_s(t) / (2 * math.pi * ell * ell)
        gamma = lambda t: 2 * lhy_coefficient(a_s(t), a_dd) / (5 * math.pi ** 1.5 * ell ** 3)
    else:
        g = 4 * math.pi * a_s / (2 * math.pi * ell * ell)
        gamma = 2 * lhy_coefficient(a_s, a_dd) / (5 * math.pi ** 1.5 * ell ** 3)
    terms = [Contact(g), LHY(gamma), TubeDipolar1D(grid, 4 * math.pi * a_dd, ell, eta, kernel)]
    if potential is not None:
        terms.insert(0, Potential(potential))
    return GPE(grid, *terms)

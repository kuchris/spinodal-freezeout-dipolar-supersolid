"""Physics terms of the Hamiltonian.

The kinetic term -1/2 nabla^2 is built into the model and applied exactly in
k-space. Every other term here is *local*: it acts as a real potential U(x, t)
that multiplies psi, so its split step is a pointwise phase exp(-i U tau).
For the terms below, |psi| does not change during a real-time local step, so
that step is exact (for a time-dependent potential, up to evaluating V at the
substep time, which keeps the scheme second order).

A local term provides (``nonlinear`` marks terms that depend on the density)
    local(density, t)          -> real potential U, broadcastable to the field
    energy_density(density, t) -> real energy density whose integral is its energy
"""

from __future__ import annotations

import functools
import math

import torch



def g_fft(grid, f):
    return torch.fft.fftn(f, dim=grid.dims)


def g_ifft(grid, f):
    return torch.fft.ifftn(f, dim=grid.dims)


class Potential:
    """External potential: a tensor V(x), or a callable t -> V(x, t)."""

    nonlinear = False
    energy_fraction = 1.0   # energy density = energy_fraction * local * density

    def __init__(self, V):
        self.V = V

    def at(self, t):
        return self.V(t) if callable(self.V) else self.V

    def local(self, density, t=0.0):
        return self.at(t)

    def response(self, density, drho, q=0.0, axis=-1):
        return 0.0

    def energy_density(self, density, t=0.0):
        return self.at(t) * density


def _strength(value):
    """A constant, or a callable t -> value (e.g. a scattering-length ramp)."""
    return value if callable(value) else float(value)


class Contact:
    """Contact interaction g |psi|^2; ``g`` may be a callable t -> g(t) (ramps of
    the scattering length; the split step evaluates it at each local substep's
    time, ground states and energies at their ``t``)."""

    nonlinear = True
    linear_in_density = True    # U(a rho1 + b rho2) = a U(rho1) + b U(rho2)
    energy_fraction = 0.5

    def __init__(self, g):
        self._g = _strength(g)

    def at(self, t):
        return self._g(t) if callable(self._g) else self._g

    @property
    def g(self):
        """g at t = 0."""
        return self.at(0.0)

    def local(self, density, t=0.0):
        return self.at(t) * density

    def response(self, density, drho, q=0.0, axis=-1):
        """Linear change of the potential for a density change drho (Bogoliubov, at t = 0)."""
        return self.g * drho

    def energy_density(self, density, t=0.0):
        return 0.5 * self.at(t) * density * density


def harmonic(grid, omega=1.0, center=None):
    """Harmonic trap 1/2 sum_i omega_i^2 (x_i - c_i)^2 as a full spatial array."""
    omegas = (omega,) * grid.ndim if isinstance(omega, (int, float)) else tuple(omega)
    centers = (0.0,) * grid.ndim if center is None else tuple(center)
    return sum(0.5 * w * w * (x - c) ** 2 for w, x, c in zip(omegas, grid.x, centers))


def gaussian_obstacle(grid, height, width, center):
    """Gaussian bump height * exp(-|x - c(t)|^2 / width^2) as a callable t -> V.

    ``center`` is a callable t -> coordinates (a moving stirrer) or fixed
    coordinates. Distances use the nearest periodic image.
    """
    path = center if callable(center) else (lambda t, c=tuple(center): c)

    def V(t):
        r2 = 0.0
        for x, L, c in zip(grid.x64, grid.length, path(t)):
            d = torch.remainder(x - c + L / 2, L) - L / 2
            r2 = r2 + d * d
        return (height * torch.exp(-r2 / width ** 2)).to(grid.real_dtype)

    return V


class Twist:
    """Phase twist (Bloch quasi-momentum) q along one axis, for ground states.

    Writing psi = exp(i q x) u with u periodic turns the kinetic energy of u
    into 1/2 (p + q)^2 = 1/2 p^2 + q p + q^2 / 2; this term is the q p + q^2 / 2
    part, applied spectrally. The ground state of the twisted model gives the
    superfluid density from E(q) - E(0) = rho_s q^2 / 2 per unit length (Leggett):
    a uniform superfluid has rho_s = rho, a crystal of isolated droplets 0.
    Only :func:`dipgpe.ground_state` supports it; SplitStep refuses it.
    """

    nonlinear = False

    def __init__(self, q, axis=-1):
        self.q, self.axis = float(q), int(axis)

    def apply(self, grid, psi):
        return self.apply_k(grid, psi, grid.fft(psi))

    def apply_k(self, grid, psi, psi_k):
        """apply() given psi_k = fft(psi), which callers often have already."""
        k = grid.k[self.axis]
        return grid.ifft((self.q * k + 0.5 * self.q ** 2) * psi_k)

    def energy(self, grid, psi):
        return grid.integrate((psi.conj() * self.apply(grid, psi)).real)


class Rotation:
    """Rotating frame about the z axis: the term -Omega L_z, L_z = -i (x d_y - y d_x).

    Not a local term: the split step applies it exactly inside the kinetic step
    by writing -1/2 nabla^2 - Omega L_z as
        1/2 (p_x + Omega y)^2 + 1/2 (p_y - Omega x)^2 - Omega^2 (x^2 + y^2) / 2,
    where each square is diagonal in one k direction at fixed position and the
    last (centrifugal) part joins the local step. Every kinetic piece is
    non-negative, so imaginary time stays stable. Needs a 2D or 3D grid.
    """

    nonlinear = False

    def __init__(self, omega):
        self.omega = float(omega)

    def apply(self, grid, psi):
        """-Omega L_z psi, with spectral derivatives."""
        return self.apply_k(grid, psi, grid.fft(psi))

    def apply_k(self, grid, psi, psi_k):
        """apply() given psi_k = fft(psi), which callers often have already."""
        d_x = grid.ifft(1j * grid.k[0] * psi_k)
        d_y = grid.ifft(1j * grid.k[1] * psi_k)
        return (1j * self.omega) * (grid.x[0] * d_y - grid.x[1] * d_x)

    def energy(self, grid, psi):
        """-Omega <L_z> (not divided by N)."""
        return grid.integrate((psi.conj() * self.apply(grid, psi)).real)


class Dipolar:
    """Dipole-dipole interaction of dipoles polarized along unit vector ``e``.

    Phi_dd = U_dd * |psi|^2 (convolution), evaluated with real FFTs using
        U_dd(k) = g_dd (3 (k.e)^2 / k^2 - 1),   g_dd = 4 pi a_dd (hbar = m = 1),
    which is the kernel g_dd (3 / 4 pi) (1 - 3 cos^2 theta) / r^3. The FFT makes the
    interaction periodic. ``cutoff="sphere"`` truncates it at radius R (default:
    half the smallest box length) with the factor
        1 + 3 cos(kR) / (kR)^2 - 3 sin(kR) / (kR)^3          (Ronen et al. 2006),
    which equals the real-space kernel cut at r = R. Periodic images then drop out
    if the cloud's diameter D satisfies D < R and D < L - R (for R = L/2: the box
    must be at least twice the cloud's size). ``cutoff="tube"`` keeps the system
    periodic along z (an infinite tube) and cuts the kernel only at transverse
    distance rho = R (default half the smaller transverse box length); dipoles
    must lie in the x-y plane (see :func:`_tube_outer`). Without it, transverse
    images of a tube interact through 1/d^2 line-line forces and the energy
    converges only as 1/L_perp^2 (measured: 4.23 -> 5.95 per atom for L_perp
    8 -> 32 on a 164Dy tube). ``cutoff=None``
    keeps the periodic sum (e.g. for a tube that is meant to be periodic along
    one axis, with transverse images checked by enlarging the box).
    The k = 0 component is set to 0 in both cases. ``cutoff="cylinder"`` (dipoles
    along z) cuts the kernel outside |z| <= Z, rho <= R (defaults: half the box
    height, half the smaller transverse length), so a flat cloud needs a box only
    about twice its height along z: the kernel is the full one minus the slab
    |z| > Z (analytic: -3 g_dd exp(-k_rho Z) k_rho (k_rho cos k_z Z - k_z sin k_z Z) / k^2)
    minus the annulus rho > R, |z| <= Z (quadrature, see _cylinder_annulus); its
    k = 0 value is g_dd (2 - 3 Z / sqrt(R^2 + Z^2)). What matters are pair
    separations: a cloud of radius a and full height H keeps all its own
    interactions and none with images if 2a < R, 2a < L - R, H < Z, H < L_z - Z,
    i.e. a < L/4 and H < L_z/2 for the defaults.
    """

    nonlinear = True
    linear_in_density = True
    energy_fraction = 0.5

    def __init__(self, grid, g_dd, direction=(0.0, 0.0, 1.0), cutoff="sphere", radius=None, height=None):
        if grid.ndim != 3:
            raise ValueError("Dipolar needs a 3D grid")
        e = torch.tensor(direction, dtype=torch.float64)
        self.e = (e / e.norm()).tolist()
        self.grid, self.g_dd = grid, float(g_dd)
        if cutoff not in ("sphere", "tube", "cylinder", None):
            raise ValueError("cutoff must be 'sphere', 'tube', 'cylinder' or None")
        if cutoff == "tube" and abs(self.e[2]) > 1e-12:
            raise NotImplementedError("tube cutoff needs dipoles perpendicular to the tube axis z")
        if cutoff == "cylinder" and abs(abs(self.e[2]) - 1) > 1e-12:
            raise NotImplementedError("cylinder cutoff needs dipoles along the cylinder axis z")
        self.cutoff = cutoff
        self.radius = radius if radius is not None else (
            min(grid.length) / 2 if cutoff == "sphere" else min(grid.length[:2]) / 2)
        self.height = height if height is not None else grid.length[2] / 2
        k = [grid.k64[0], grid.k64[1],
             (2 * torch.pi * torch.fft.rfftfreq(grid.shape[2], d=grid.dx[2], dtype=torch.float64)
              ).reshape(1, 1, -1).to(grid.device)]
        self.kernel = self._kernel(k).to(grid.real_dtype)
        self._shifted = {}

    def _kernel(self, k):
        """The (cut-off) kernel on broadcastable float64 wavevector components k."""
        e, R = self.e, self.radius
        k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
        k2_safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
        ke = e[0] * k[0] + e[1] * k[1] + e[2] * k[2]
        kernel = self.g_dd * (3 * ke * ke / k2_safe - 1)
        zero = torch.zeros_like(kernel)
        if self.cutoff == "sphere":
            kr = torch.sqrt(k2_safe) * R
            kernel = kernel * (1 + 3 * torch.cos(kr) / kr ** 2 - 3 * torch.sin(kr) / kr ** 3)
        elif self.cutoff == "tube":
            kernel = kernel - _tube_outer(k, e, R, self.g_dd)
            zero = torch.full_like(kernel, self.g_dd / 2)  # k -> 0 limit of the truncated kernel
        elif self.cutoff == "cylinder":
            Z = self.height
            krho = torch.sqrt(k[0] ** 2 + k[1] ** 2)
            kz = k[2].abs()
            slab = -3 * self.g_dd * torch.exp(-krho * Z) * krho * (krho * torch.cos(kz * Z) - kz * torch.sin(kz * Z)) \
                / k2_safe
            kernel = kernel - slab - _cylinder_annulus(krho, kz, R, Z, self.g_dd)
            zero = torch.full_like(kernel, self.g_dd * (2 - 3 * Z / math.sqrt(R * R + Z * Z)))
        return torch.where(k2 > 0, kernel, zero)

    def response(self, density, drho, q=0.0, axis=-1):
        """Phi_dd of a (complex) density change drho = exp(i q x_axis) * periodic,
        returned without the exp(i q x_axis) factor (Bloch form, for Bogoliubov)."""
        key = (float(q), axis % 3)
        if key not in self._shifted:
            g = self.grid
            k = [g.k64[0].clone(), g.k64[1].clone(), g.k64[2].clone()]
            k[key[1]] = k[key[1]] + key[0]
            self._shifted[key] = self._kernel(k).to(g.real_dtype)
        return g_ifft(self.grid, g_fft(self.grid, drho) * self._shifted[key])

    def potential(self, density):
        """Phi_dd for a real density (spatial axes last, leading axes allowed)."""
        dims = self.grid.dims
        return torch.fft.irfftn(torch.fft.rfftn(density, dim=dims) * self.kernel,
                                s=self.grid.shape, dim=dims)

    def local(self, density, t=0.0):
        return self.potential(density)

    def energy_density(self, density, t=0.0):
        return 0.5 * self.potential(density) * density


def _tube_outer(k, e, R, g_dd, panel=0.5, nodes_per_panel=16):
    """Fourier transform of the dipolar kernel restricted to rho > R (all z).

    Dipoles along e = (cos a, sin a, 0), tube axis z, q = |k_z|. With
    U = (3 g_dd / 4 pi) (1/r^3 - 3 (e.r)^2 / r^5), the z transform at transverse
    position (rho, phi) is (3 g_dd / 4 pi) [A(rho) - q^2 K2(q rho) cos 2(phi - a)],
        A = (2 q / rho) K1(q rho) - q^2 K2(q rho),
    and the angular transform gives
        outer = (3 g_dd / 2) int_R^inf rho [A J0(k rho) + q^2 K2(q rho) J2(k rho) cos 2(phi_k - a)] d rho.
    At q = 0, A = 0 and the J2 integral is analytic: outer = 3 g_dd J1(kR)/(kR) cos 2(phi_k - a).
    For q > 0 the integrand decays like exp(-q rho); composite Gauss-Legendre on
    [R, R + 40 / q_min] in panels of ``panel`` resolves the Bessel oscillations.
    """
    kx, ky, kz = k
    device = kx.device
    alpha = math.atan2(e[1], e[0])
    kperp = torch.sqrt(kx ** 2 + ky ** 2)                       # (Nx, Ny, 1)
    phi_k = torch.atan2(ky, kx)
    cos2 = torch.cos(2 * (phi_k - alpha))
    q_all = kz.reshape(-1).abs()                                 # any kz grid (rfft, full, Bloch-shifted)
    q, q_index = torch.unique(q_all, return_inverse=True)        # distinct |kz|, ascending
    shape = (kperp.shape[0], kperp.shape[1], q.numel())
    outer = torch.zeros(shape, dtype=torch.float64, device=device)

    x = kperp * R
    j1_over = torch.where(x > 0, torch.special.bessel_j1(x) / torch.where(x > 0, x, 1.0), 0.5)
    zero = q == 0
    if bool(zero.any()):                                         # q = 0: analytic
        outer[:, :, zero] = (3 * g_dd * j1_over * cos2).expand(-1, -1, int(zero.sum()))

    pos = q > 0
    if bool(pos.any()):
        q_pos = q[pos]
        span = 40.0 / float(q_pos.min())
        n_panels = max(1, math.ceil(span / panel))
        u, w = _gauss_legendre(nodes_per_panel)
        u, w = u.to(device), w.to(device)
        starts = R + panel * torch.arange(n_panels, dtype=torch.float64, device=device)
        rho = (starts[:, None] + panel * (u[None, :] + 1) / 2).reshape(-1)       # (M,)
        wr = (panel / 2 * w).repeat(n_panels) * rho                              # weights * rho
        qr = q_pos[None, :] * rho[:, None]                                       # (M, Nq)
        k0, k1 = torch.special.modified_bessel_k0(qr), torch.special.modified_bessel_k1(qr)
        k2 = k0 + 2 * k1 / qr
        A = (2 * q_pos[None, :] / rho[:, None]) * k1 - q_pos[None, :] ** 2 * k2  # (M, Nq)
        B = q_pos[None, :] ** 2 * k2
        flat_k = kperp.reshape(-1)
        unique_k, inverse = torch.unique(flat_k, return_inverse=True)
        IA = torch.empty((unique_k.numel(), q_pos.numel()), dtype=torch.float64, device=device)
        IB = torch.empty_like(IA)
        for start in range(0, unique_k.numel(), 512):            # bound memory: (512, M) at a time
            kr = unique_k[start:start + 512, None] * rho[None, :]
            j0, j1 = torch.special.bessel_j0(kr), torch.special.bessel_j1(kr)
            j2 = torch.where(kr > 0, 2 * j1 / torch.where(kr > 0, kr, 1.0) - j0, 0.0)
            IA[start:start + 512] = (j0 * wr) @ A
            IB[start:start + 512] = (j2 * wr) @ B
        IA = IA[inverse].reshape(shape[0], shape[1], -1)
        IB = IB[inverse].reshape(shape[0], shape[1], -1)
        outer[:, :, pos] = 1.5 * g_dd * (IA + IB * cos2)
    outer = outer[:, :, q_index]                                 # back to the kz grid order
    if kz.dim() == 3 and kz.shape[2] == 1 and kz.shape[0] > 1:   # kz along another axis is not supported
        raise NotImplementedError("tube cutoff needs the tube axis as the last grid axis")
    return outer



def _cylinder_annulus(krho, kz, R, Z, g_dd, panel=0.5, nodes_per_panel=16):
    """Fourier transform of the z-dipole kernel restricted to rho > R, |z| <= Z.

    With int_0^inf rho J0(k_rho rho) (rho^2 - 2 z^2) / (rho^2 + z^2)^(5/2) d rho
    = -k_rho exp(-k_rho z) and int_0^R rho (rho^2 - 2 z^2) / (rho^2 + z^2)^(5/2) d rho
    = -R^2 / (R^2 + z^2)^(3/2), the annulus is (no truncated tail)
        3 g_dd int_0^Z cos(k_z z) [-k_rho exp(-k_rho z) + R^2 / (R^2 + z^2)^(3/2) - B(z)] dz,
        B(z) = int_0^R rho (J0(k_rho rho) - 1) (rho^2 - 2 z^2) / (rho^2 + z^2)^(5/2) d rho,
    whose integrand is regular at rho, z -> 0 (J0 - 1 ~ rho^2). The exp(-k_rho z)
    term is integrated analytically (it is steep for large k_rho); the rest by
    composite Gauss-Legendre with panels of ``panel``."""
    device = krho.device
    ukr, ikr = torch.unique(krho.reshape(-1), return_inverse=True)
    ukz, ikz = torch.unique(kz.reshape(-1), return_inverse=True)
    up, wp = _gauss_legendre(nodes_per_panel)
    up, wp = up.to(device), wp.to(device)

    def nodes(a, b):
        """Panels of ``panel`` on [a, b]; the first one graded geometrically towards a
        (ratio 1/4, 12 levels): near rho, z -> 0 the integrand is bounded but not smooth
        (its limit depends on the direction of approach)."""
        n = max(1, math.ceil((b - a) / panel))
        h = (b - a) / n
        edges = [a + h * 4.0 ** -j for j in range(12, 0, -1)] + [a + h * i for i in range(1, n + 1)]
        edges = torch.tensor([a] + edges, dtype=torch.float64, device=device)
        lo, hi = edges[:-1], edges[1:]
        x = (lo[:, None] + (hi - lo)[:, None] * (up[None, :] + 1) / 2).reshape(-1)
        w = ((hi - lo)[:, None] / 2 * wp[None, :]).reshape(-1)
        return x, w

    z, wz = nodes(0.0, Z)
    rho, wr = nodes(0.0, R)
    K = (rho[:, None] ** 2 - 2 * z[None, :] ** 2) / (rho[:, None] ** 2 + z[None, :] ** 2) ** 2.5        # (Mr, Mz)
    out = torch.empty((ukr.numel(), ukz.numel()), dtype=torch.float64, device=device)
    cos = torch.cos(z[:, None] * ukz[None, :])                                                       # (Mz, Nkz)
    base = R * R / (R * R + z * z) ** 1.5                                                            # (Mz,)
    kz_ = ukz[None, :]
    for start in range(0, ukr.numel(), 256):
        kr = ukr[start:start + 256, None]
        B = ((torch.special.bessel_j0(kr * rho[None, :]) - 1) * (wr * rho)[None, :]) @ K           # (nk, Mz)
        # -k_rho int_0^Z cos(k_z z) exp(-k_rho z) dz, analytic (steep for large k_rho)
        k2 = kr * kr + kz_ * kz_
        expo = torch.where(k2 > 0, (kr - torch.exp(-kr * Z) * (kr * torch.cos(kz_ * Z) - kz_ * torch.sin(kz_ * Z)))
                           / torch.where(k2 > 0, k2, 1.0), torch.full_like(k2, Z))
        out[start:start + 256] = -kr * expo + ((base[None, :] - B) * wz[None, :]) @ cos
    return 3 * g_dd * out[ikr.reshape(krho.shape), ikz.reshape(kz.shape)]

class LHY:
    """Lee-Huang-Yang quantum-fluctuation term gamma |psi|^3 (energy (2/5) gamma n^(5/2))."""

    nonlinear = True
    energy_fraction = 0.4

    def __init__(self, gamma):
        self._gamma = _strength(gamma)    # a constant or a callable t -> gamma(t)

    def at(self, t):
        return self._gamma(t) if callable(self._gamma) else self._gamma

    @property
    def gamma(self):
        """gamma at t = 0."""
        return self.at(0.0)

    def local(self, density, t=0.0):
        return self.at(t) * density ** 1.5

    def response(self, density, drho, q=0.0, axis=-1):
        return 1.5 * self.gamma * density ** 0.5 * drho

    def energy_density(self, density, t=0.0):
        return 0.4 * self.at(t) * density ** 2.5


def q5(eps_dd, points=2000):
    """Q5(eps) = Re int_0^1 (1 - eps + 3 eps u^2)^(5/2) du (Lima & Pelster); the
    integrand's real part vanishes where the base is negative (eps > 1)."""
    u, w = _gauss_legendre(points)
    u, w = 0.5 * (u + 1), 0.5 * w
    base = 1 - eps_dd + 3 * eps_dd * u * u
    return float((w * torch.clamp(base, min=0.0) ** 2.5).sum())


def lhy_coefficient(a_s, a_dd):
    """gamma_QF = (32/3) g sqrt(a_s^3 / pi) Q5(a_dd / a_s), g = 4 pi a_s (hbar = m = 1)."""
    return 32.0 / 3.0 * (4 * torch.pi * a_s) * (a_s ** 3 / torch.pi) ** 0.5 * q5(a_dd / a_s)


def _gauss_legendre(n):
    """Nodes and weights on [-1, 1] (Golub-Welsch), float64 (copies of a cache:
    q5 is called at every substep of a scattering-length ramp)."""
    nodes, weights = _gauss_legendre_cached(n)
    return nodes.clone(), weights.clone()


@functools.lru_cache(maxsize=None)
def _gauss_legendre_cached(n):
    i = torch.arange(1, n, dtype=torch.float64)
    beta = i / torch.sqrt(4 * i * i - 1)
    J = torch.diag(beta, 1) + torch.diag(beta, -1)
    nodes, vectors = torch.linalg.eigh(J)
    return nodes, 2 * vectors[0] ** 2

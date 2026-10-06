"""Split-step Fourier time stepping.

One step of a scheme is the composition

    L(a0) K(b0) L(a1) K(b1) ... K(b_{m-1}) L(a_m)

where K(b) = exp(-i b tau T) is the exact kinetic step in k-space and
L(a) = exp(-i a tau U(|psi|^2)) the pointwise local step. Real time uses
tau = dt; imaginary time uses tau = -i dt and renormalizes after each step.
In real time the last local substep of a step is merged with the first one
of the next step, so Strang costs one FFT pair per step.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from .diagnostics import density, norm


@dataclass(frozen=True)
class Scheme:
    name: str
    a: tuple  # local-step coefficients, length m + 1
    b: tuple  # kinetic-step coefficients, length m
    order: int


STRANG = Scheme("strang", (0.5, 0.5), (1.0,), 2)

# Yoshida's fourth-order triple jump of Strang steps. The middle weight is
# negative, which is fine in real time but diverges in imaginary time.
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = 1.0 - 2.0 * _W1
YOSHIDA4 = Scheme("yoshida4",
                  (_W1 / 2, (_W1 + _W0) / 2, (_W0 + _W1) / 2, _W1 / 2),
                  (_W1, _W0, _W1), 4)


def max_stable_dt(grid, scheme=STRANG, safety=0.8):
    """Largest real-time dt that avoids the split-step resonance instability.

    For the nonlinear GPE on a non-vanishing background, a split step is
    unstable once the kinetic phase of the highest retained mode,
    |b| dt |k|^2_max / 2 (plus g n dt), reaches ~pi: the discrete kinetic rotation
    resonates with the nonlinear step (cf. Weideman & Herbst 1986). Measured on a
    2D uniform condensate: stable at 0.92 of the bound, unstable at 0.97. In 2D on
    a square grid the bound is dt < dx^2 / pi, so the step is limited by dx^2 like
    explicit schemes. Returns safety * 2 pi / (max|b| |k|^2_max).

    ``grid`` may also be a model. With a :class:`dipgpe.terms.Rotation` the bound is
    unchanged for trapped clouds: measured at Omega = 0.5 and 0.9 (2D, g = 100),
    stable at 0.95 and unstable at 1.1 of it, as without rotation, because the
    kinetic pieces 1/2 (k_x + Omega y)^2 only exceed k_max^2/2 where the density
    vanishes. A rotating state with density at large |r| may need a smaller dt.
    """
    grid = getattr(grid, "grid", grid)
    k2_max = sum((math.pi / dx) ** 2 for dx in grid.dx)
    return safety * 2 * math.pi / (max(abs(b) for b in scheme.b) * k2_max)


class SplitStep:
    """Advances psi under a :class:`dipgpe.model.GPE` with a fixed dt.

    The stepper keeps the physical time ``t`` (start ``t0``), advanced by each
    real-time :meth:`evolve` call. A local substep is evaluated at the time the
    kinetic substeps have reached, so time-dependent potentials stay second order
    (Strang) and the merged boundary substeps share one time.

    keep_norm (real time only): rescale psi to its input norm at the end of each
    :meth:`evolve` call. The GPE conserves the norm exactly, but single-precision
    cuFFT loses about 1e-7 to 4e-7 of it per FFT pair (measured on an RTX 5070 Ti),
    which accumulates linearly in long complex64 runs. This projection removes
    that drift; it does not correct phase round-off. For long runs call evolve in
    chunks (tens to hundreds of steps): drift inside one long call still shifts
    the nonlinear phase rotation.

    A real-time stepper for a nonlinear model warns when dt exceeds
    max_stable_dt(safety=0.9): such runs blow up after a growth time of order
    1 / (g n) (see :func:`max_stable_dt`).
    """

    def __init__(self, model, dt, scheme=STRANG, *, imaginary=False, keep_norm=False, t0=0.0):
        if imaginary and min(scheme.a + scheme.b) < 0:
            raise ValueError(f"{scheme.name} has negative substeps; use STRANG in imaginary time")
        self.model = model
        self.dt = float(dt)
        self.scheme = scheme
        self.imaginary = imaginary
        self.keep_norm = keep_norm
        self.t = float(t0)
        self.tau = -1j * self.dt if imaginary else complex(self.dt)
        grid = model.grid
        self.omega = model.rotation
        if self.omega:
            self._rot_x = {b: self._rotation_factor_x(b / 2) for b in set(scheme.b)}
            self._rot_yz = {b: self._rotation_factor_yz(b) for b in set(scheme.b)}
            self._centrifugal = (-0.5 * self.omega ** 2 * (grid.x64[0] ** 2 + grid.x64[1] ** 2)
                                 ).to(grid.real_dtype)
        else:
            self._kinetic = {b: self._kinetic_factor(b) for b in set(scheme.b)}
        unsupported = [type(t).__name__ for t in model.terms if not hasattr(t, "local") and not hasattr(t, "omega")]
        if unsupported:
            raise NotImplementedError(f"SplitStep cannot apply {', '.join(unsupported)} (ground_state only)")
        nonlinear = any(getattr(term, "nonlinear", True) for term in model.terms)
        limit = max_stable_dt(model, scheme, safety=0.9)
        if not imaginary and nonlinear and self.dt > limit:
            warnings.warn(f"dt = {self.dt:.3g} exceeds the split-step stability limit "
                          f"~{limit:.3g} for this grid; long nonlinear runs will blow up "
                          f"(use max_stable_dt)", RuntimeWarning, stacklevel=2)
        offsets = [0.0]
        for b in scheme.b:
            offsets.append(offsets[-1] + b)
        self._offsets = offsets  # kinetic time reached before local substep j, in units of dt

    def _kinetic_factor(self, b):
        """exp(-i b tau k^2 / 2), built per axis in float64 so phases stay accurate."""
        grid = self.model.grid
        factor = None
        for ki in grid.k64:
            f = torch.exp((-0.5j * b * self.tau) * ki * ki).to(grid.dtype)
            factor = f if factor is None else factor * f
        return factor

    def _rotation_factor_x(self, c):
        """exp(-i c tau (k_x + Omega y)^2 / 2) in (k_x, y) space; constant along z."""
        grid = self.model.grid
        phase = 0.5 * (grid.k64[0] + self.omega * grid.x64[1]) ** 2
        return torch.exp((-1j * c * self.tau) * phase).to(grid.dtype)

    def _rotation_factor_yz(self, c):
        """exp(-i c tau [(k_y - Omega x)^2 + k_z^2] / 2) in (x, k_y, k_z) space."""
        grid = self.model.grid
        phase = 0.5 * (grid.k64[1] - self.omega * grid.x64[0]) ** 2
        if grid.ndim == 3:
            phase = phase + 0.5 * grid.k64[2] ** 2
        return torch.exp((-1j * c * self.tau) * phase).to(grid.dtype)

    def _local(self, psi, a, t):
        if a == 0.0:
            return psi
        U = self.model.local_potential(density(psi), t)
        if self.omega:
            U = U + self._centrifugal
        if not torch.is_tensor(U):
            return psi
        return psi * torch.exp((-1j * a * self.tau) * U)

    def _kinetic_step(self, psi, b):
        grid = self.model.grid
        if not self.omega:
            return grid.ifft(grid.fft(psi) * self._kinetic[b])
        # Symmetric split X(b/2) YZ(b) X(b/2) of the rotating-frame kinetic step.
        x_dim, yz_dims = grid.dims[0], grid.dims[1:]
        psi = torch.fft.ifft(torch.fft.fft(psi, dim=x_dim) * self._rot_x[b], dim=x_dim)
        psi = torch.fft.ifftn(torch.fft.fftn(psi, dim=yz_dims) * self._rot_yz[b], dim=yz_dims)
        return torch.fft.ifft(torch.fft.fft(psi, dim=x_dim) * self._rot_x[b], dim=x_dim)

    def _renormalize(self, psi, target):
        grid = self.model.grid
        scale = torch.sqrt(target / norm(grid, psi)).to(grid.real_dtype)
        return psi * scale.reshape(scale.shape + (1,) * grid.ndim)

    def evolve(self, psi, steps):
        """Return psi advanced by ``steps`` steps of size dt (real time also advances t)."""
        a, b = self.scheme.a, self.scheme.b
        if steps <= 0:
            return psi
        if self.imaginary:
            # Renormalize before every local substep, so the nonlinear terms
            # always see the target amplitude. Renormalizing only once per step
            # leaves a first-order bias of relative size ~ mu * dt in the fixed
            # point (measured: 1.3% virial residual at mu*dt = 0.04).
            target = norm(self.model.grid, psi)
            for _ in range(steps):
                psi = self._local(psi, a[0], self.t)
                for j, bj in enumerate(b):
                    psi = self._renormalize(self._kinetic_step(psi, bj), target)
                    psi = self._local(psi, a[j + 1], self.t)
                psi = self._renormalize(psi, target)
            return psi
        target = norm(self.model.grid, psi) if self.keep_norm else None
        dt, offsets, last = self.dt, self._offsets, len(b) - 1
        psi = self._local(psi, a[0], self.t)
        for s in range(steps):
            t_step = self.t + s * dt
            for j, bj in enumerate(b):
                coeff = a[j + 1] + (a[0] if j == last and s < steps - 1 else 0.0)
                psi = self._local(self._kinetic_step(psi, bj), coeff, t_step + offsets[j + 1] * dt)
        self.t += steps * dt
        return psi if target is None else self._renormalize(psi, target)

"""Ground states by direct energy minimization: preconditioned nonlinear
conjugate gradient on the sphere ||psi||^2 = N.

Imaginary time is a (split-step) gradient flow: its step count grows like the
condition number of the energy Hessian, which diverges near a continuous
transition and near the end of a metastable branch (critical slowing, ISSUES
P4-5). Conjugate gradient needs roughly its square root. The method follows
Antoine, Levitt & Tang, J. Comput. Phys. 343, 92 (2017): Riemannian gradient
r = H psi - lambda psi, kinetic preconditioner (alpha + k^2/2)^-1 made tangent
to the sphere, Polak-Ribiere(+) conjugation, and the exact retraction
psi(theta) = cos(theta) psi + sin(theta) q (||q||^2 = N), so the norm is kept
to round-off and the minimizer carries no time-step bias.
"""

from __future__ import annotations

import math

import torch

from .diagnostics import density, norm


def _expand(s, grid):
    return s.to(grid.real_dtype).reshape(s.shape + (1,) * grid.ndim)


def _inner(grid, a, b):
    """Re integral conj(a) b, leading axes kept."""
    return grid.integrate(a.real * b.real + a.imag * b.imag)


def energy_and_gradient(model, psi, t=0.0, gradient=True):
    """(E, H psi) with E the energy (not divided by N) and H psi = dE/dpsi*.

    One forward FFT serves the kinetic energy and -1/2 nabla^2 psi; each local
    term's potential is computed once (terms with ``energy_fraction`` c give
    their energy as c * integral U rho without a second evaluation)."""
    grid = model.grid
    psi_k = grid.fft(psi)
    E = 0.5 * (grid.k2 * density(psi_k)).sum(dim=grid.dims, dtype=torch.float64) * (grid.dV / grid.size)
    rho = density(psi)
    U = 0.0
    for term in model.local_terms:
        u = term.local(rho, t)
        c = getattr(term, "energy_fraction", None)
        E = E + (c * grid.integrate(u * rho) if c is not None else grid.integrate(term.energy_density(rho, t)))
        U = U + u
    h = None
    if gradient:
        h = grid.ifft(0.5 * grid.k2 * psi_k)
        if torch.is_tensor(U):
            h = h + U * psi
    for term in model.terms:
        if not hasattr(term, "local"):            # linear non-local terms (rotation)
            a = term.apply_k(grid, psi, psi_k) if hasattr(term, "apply_k") else term.apply(grid, psi)
            E = E + _inner(grid, psi, a)
            if gradient:
                h = h + a
    return E, h


class _Geodesic:
    """Energy and gradient along psi(theta) = c psi + s q (c = cos theta, s = sin theta)
    without applying FFT-based operators to psi(theta) itself: A = -1/2 nabla^2 plus
    the non-local terms is linear, A psi(theta) = c A psi + s A q, and the potential
    L of the terms linear in the density (contact, dipolar) combines as
    c^2 L(|psi|^2) + s^2 L(|q|^2) + 2 c s L(Re psi* q). Only q needs transforms
    (once per iteration, for any number of trial angles); the other local terms
    (potential, LHY) are pointwise. A psi and L(|psi|^2) of the accepted state are
    carried to the next iteration and recomputed exactly every few iterations."""

    def __init__(self, model, t):
        self.grid, self.t = model.grid, t
        self.linear = [m for m in model.local_terms if getattr(m, "linear_in_density", False)
                       and getattr(m, "energy_fraction", None) == 0.5]
        self.other = [m for m in model.local_terms if all(m is not n for n in self.linear)]
        self.ops = [m for m in model.terms if not hasattr(m, "local")]

    def apply(self, f):
        g = self.grid
        f_k = g.fft(f)
        a = g.ifft(0.5 * g.k2 * f_k)
        for op in self.ops:
            a = a + (op.apply_k(g, f, f_k) if hasattr(op, "apply_k") else op.apply(g, f))
        return a

    def potential(self, rho):
        U = 0.0
        for m in self.linear:
            U = U + m.local(rho, self.t)
        return U

    def state(self, psi):
        """(A psi, L(|psi|^2)) computed from scratch."""
        return self.apply(psi), self.potential(density(psi))

    def direction(self, psi, q):
        return self.apply(q), self.potential(density(q)), self.potential(psi.real * q.real + psi.imag * q.imag)

    @staticmethod
    def along(state, dirn, c, s):
        (A, L), (Aq, Lq, Lx) = state, dirn
        return A * c + Aq * s, (L * (c * c) + Lq * (s * s) + Lx * (2 * c * s) if torch.is_tensor(L) else L)

    def evaluate(self, psi, state):
        """(E, H psi) from psi and its (A psi, L)."""
        g = self.grid
        A, L = state
        rho = density(psi)
        E = _inner(g, psi, A)
        U = L
        if torch.is_tensor(L):
            E = E + 0.5 * g.integrate(L * rho)
        for m in self.other:
            u = m.local(rho, self.t)
            c = getattr(m, "energy_fraction", None)
            E = E + (c * g.integrate(u * rho) if c is not None else g.integrate(m.energy_density(rho, self.t)))
            U = U + u
        return E, (A + U * psi if torch.is_tensor(U) or U != 0 else A)


def _pick(mask, a, b):
    """torch.where over (possibly nested tuples of) tensors; non-tensors pass through."""
    if isinstance(a, tuple):
        return tuple(_pick(mask, x, y) for x, y in zip(a, b))
    return torch.where(mask, a, b) if torch.is_tensor(a) else a


def ground_state(model, psi, *, tol=None, max_iter=200_000, max_angle=0.05, check_every=50, live=None, t=0.0,
                 strict=True, preconditioner="kinetic", callback=None, callback_every=1000, reuse=True):
    """Minimize E[psi] at fixed norm, starting from ``psi`` (its norm is kept).

    Converges to the local minimum whose basin contains ``psi`` (metastable
    states included). Stops when the eigenstate residual
    ||H psi - mu psi|| / ||psi|| (see :func:`dipgpe.residual`) is below ``tol``
    for every batch entry. The default is 1e-9 (complex128) or 1e-5 (complex64),
    raised to 4x the round-off floor k^2_max eps / 2: FFT round-off at the
    highest modes is amplified by the kinetic term (measured floor 3.3e-5 for
    complex64 at 128^2, dx = 0.125, against an estimate of 3.8e-5).
    Returns (psi, info) with info = {"iterations", "mu", "residual", "energy", "converged"}.
    If tol is not reached within max_iter, raises RuntimeError, or with
    ``strict=False`` returns the last state with info["converged"] = False.
    ``live`` (a :class:`dipgpe.live.LiveRun`) receives mu, the residual and a
    preview every ``check_every`` iterations. ``preconditioner``: "kinetic",
    (|mu| + k^2/2)^-1, or "combined", P_V^1/2 P_kin P_V^1/2 with
    P_V = (|mu| + U - min U)^-1 and U the local potential of the current state
    (Antoine, Levitt & Tang 2017), refreshed every ``check_every`` iterations.
    ``callback(iteration, psi)`` is called every ``callback_every`` iterations;
    if it returns True the minimization stops there (info["stopped"] = True,
    info["converged"] = False), e.g. once a structure no longer changes.
    ``reuse`` evaluates the line search from cached operator actions (see
    :class:`_Geodesic`; same iterates up to round-off, fewer FFTs); False
    re-evaluates the full energy and gradient at every trial point.
    info["capped"] is the fraction of iterations whose step angle hit ``max_angle``.
    """
    if preconditioner not in ("kinetic", "combined"):
        raise ValueError("preconditioner must be 'kinetic' or 'combined'")
    grid = model.grid
    if tol is None:
        floor = 0.5 * sum((math.pi / dx) ** 2 for dx in grid.dx) * torch.finfo(grid.real_dtype).eps
        tol = max(1e-5 if grid.dtype == torch.complex64 else 1e-9, 4 * floor)
    # Energy comparisons tolerate round-off: float32 fields carry ~1e-7 relative
    # error into E even though it is summed in float64.
    round_off = 1e-6 if grid.dtype == torch.complex64 else 1e-12
    N = norm(grid, psi)
    sqrtN = _expand(torch.sqrt(N), grid)
    geo = _Geodesic(model, t) if reuse else None

    def full(f):
        if geo is None:
            return (*energy_and_gradient(model, f, t), None)
        st = geo.state(f)
        return (*geo.evaluate(f, st), st)

    E, h, state = full(psi)
    r_old = pg_old = p = None
    theta_t = torch.full_like(N, 0.1)
    capped = torch.zeros_like(N)
    for it in range(max_iter + 1):
        if it > 0 and it % check_every == 0:
            # Restore the norm (float32 round-off drifts it by ~1e-5 per 1000
            # iterations) and recompute E, H psi exactly (no drift from the
            # geodesic combinations either).
            psi = psi * _expand(torch.sqrt(N / _inner(grid, psi, psi)), grid)
            E, h, state = full(psi)
        n_psi = _inner(grid, psi, psi)                 # the actual norm, so r stays orthogonal to psi
        mu = _inner(grid, psi, h) / n_psi
        r = h - _expand(mu, grid) * psi
        res = torch.sqrt(_inner(grid, r, r) / n_psi)
        done = bool((res <= tol).all())
        if live is not None and (done or it % check_every == 0):
            field = psi if psi.dim() == grid.ndim else psi.reshape(-1, *grid.shape)[0]
            live.report({"iteration": it, "mu": float(mu.reshape(-1)[0]),
                         "residual": float(res.max()), "target": tol}, grid, field)
        stop = (callback is not None and it > 0 and it % callback_every == 0 and not done
                and bool(callback(it, psi)))
        if done or stop or (it == max_iter and not strict):
            return psi, {"iterations": it, "mu": mu, "residual": res, "energy": E, "converged": done,
                         "stopped": stop, "capped": capped / max(it, 1)}
        if it == max_iter:
            break
        # Preconditioned gradient, made tangent: pg = P (h - lambda_P psi).
        inv = 1.0 / (_expand(mu.abs(), grid) + 0.5 * grid.k2)
        if preconditioner == "kinetic":
            P = lambda f: grid.ifft(grid.fft(f) * inv)
        else:
            if it % check_every == 0 or it == 0:
                U = model.local_potential(density(psi), t)
                U = U - U.amin(dim=grid.dims, keepdim=True) if torch.is_tensor(U) else 0.0
                pv = (1.0 / torch.sqrt(_expand(mu.abs(), grid) + U)).to(grid.real_dtype)
            P = lambda f: pv * grid.ifft(grid.fft(pv * f) * inv)
        P_h = P(h)
        P_psi = P(psi)
        pg = P_h - _expand(_inner(grid, psi, P_h) / _inner(grid, psi, P_psi), grid) * P_psi
        if p is None:
            p = -pg
        else:
            beta = torch.clamp(_inner(grid, r - r_old, pg) / _inner(grid, r_old, pg_old), min=0.0)
            p = -pg + _expand(beta, grid) * p
            p = p - _expand(_inner(grid, psi, p) / N, grid) * psi
            restart = _inner(grid, r, p) >= 0          # not a descent direction
            p = torch.where(_expand(restart, grid).bool(), -pg, p)
        r_old, pg_old = r, pg
        q = p * (sqrtN / _expand(torch.sqrt(_inner(grid, p, p)), grid))
        slope = 2 * _inner(grid, q, r)                 # dE/dtheta at 0 (< 0)
        # Secant on the slope dE/dtheta between 0 and a trial angle. Slopes are
        # inner products, accurate to round-off relative to |r|, whereas energy
        # differences vanish below round-off (~1e-16 |E|) long before tol.
        dirn = geo.direction(psi, q) if geo is not None else None

        def at(angle):
            c, s = _expand(torch.cos(angle), grid), _expand(torch.sin(angle), grid)
            f = psi * c + q * s
            if geo is None:
                return (f, *energy_and_gradient(model, f, t), None)
            st = geo.along(state, dirn, c, s)
            return (f, *geo.evaluate(f, st), st)

        trial, E_t, h_t, state_t = at(theta_t)
        c_t, s_t = _expand(torch.cos(theta_t), grid), _expand(torch.sin(theta_t), grid)
        slope_t = 2 * _inner(grid, q * c_t - psi * s_t, h_t)
        rising = slope_t > slope
        theta = torch.where(rising, -slope * theta_t / torch.where(rising, slope_t - slope, 1.0), 2 * theta_t)
        capped = capped + (theta > max_angle).to(capped.dtype)
        theta = torch.clamp(theta, max=max_angle)
        new, E_new, h_new, state_new = at(theta)
        slack = round_off * E.abs()
        bad = (E_new > E + slack) & (E_t < E_new)      # secant step failed: take the trial point
        if bool(bad.any()):
            mask = _expand(bad, grid).bool()
            new, h_new = torch.where(mask, trial, new), torch.where(mask, h_t, h_new)
            state_new = _pick(mask, state_t, state_new)
            E_new, theta = torch.where(bad, E_t, E_new), torch.where(bad, theta_t, theta)
        worse = E_new > E + slack                      # neither lowered E: shrink and restart CG
        if bool(worse.any()):
            mask = _expand(worse, grid).bool()
            new, h_new = torch.where(mask, psi, new), torch.where(mask, h, h_new)
            state_new = _pick(mask, state, state_new)
            E_new, theta = torch.where(worse, E, E_new), torch.where(worse, theta_t / 4, theta)
            p = torch.where(mask, torch.zeros_like(p), p)
        psi, E, h, state = new, E_new, h_new, state_new
        theta_t = torch.clamp(theta, min=1e-8)
    raise RuntimeError(f"ground_state: residual {float(res.max()):.3g} > tol {tol:g} after {max_iter} iterations")

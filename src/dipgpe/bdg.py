"""Bogoliubov-de Gennes excitations of a stationary state.

For a real stationary state psi0 (chemical potential mu) the linearized eGPE
for delta psi = u exp(-i w t) - v* exp(i w t) reduces, with f_pm = u +- v, to

    w f_- = L f_+,     w f_+ = M f_-,     so  L M f_- = w^2 f_-,

where L = H[psi0] - mu (the GPE operator; L psi0 = 0 is the gauge mode) and
M = L + 2 X with X f = psi0 dU[psi0 f], the linear response of the local terms
(contact, LHY, dipolar) to the density change psi0 (u - v). For a Bloch
quasi-momentum q along one axis, u, v = exp(i q x) * periodic and the kinetic
energy becomes |k + q|^2 / 2; every operator keeps q. With M Hermitian positive
definite (a stable state; at q = 0 a crystal has translation zero modes, so use
q != 0 there) the problem is the generalized Hermitian one

    L g = w^2 M^-1 g,         f_- = M^-1 g,

solved for the lowest eigenvalues by block LOBPCG on the GPU with the kinetic
preconditioner (|k + q|^2 / 2 + s)^-1; M^-1 is applied by preconditioned
conjugate gradients. (The equivalent (M L M) g = w^2 M g is third order and
converged ~100x slower.)
"""

from __future__ import annotations

import math

import torch

from .diagnostics import chemical_potential, density


def _real_state(psi0):
    """psi0 with its global phase removed; it must then be real."""
    phase = torch.angle((psi0 * psi0).sum()) / 2
    psi = psi0 * torch.exp(-1j * phase)
    if float(psi.imag.abs().max()) > 1e-6 * float(psi.abs().max()):
        raise ValueError("bogoliubov needs a real stationary state (no rotation or current)")
    if float(psi.real.sum()) < 0:
        psi = -psi
    return psi.real.to(psi0.real.dtype)


class Bogoliubov:
    """Operators L, M (and the generalized problem) for one quasi-momentum q."""

    def __init__(self, model, psi0, q=0.0, axis=-1, t=0.0):
        grid = self.grid = model.grid
        for term in model.terms:
            if not hasattr(term, "local"):
                raise NotImplementedError(f"bogoliubov does not support {type(term).__name__}")
            if getattr(term, "nonlinear", True) and not hasattr(term, "response"):
                raise NotImplementedError(f"{type(term).__name__} has no linear response")
        self.model, self.q, self.axis, self.t = model, float(q), axis % grid.ndim, t
        self.psi0 = _real_state(psi0)
        self.rho0 = self.psi0 ** 2
        self.mu = float(chemical_potential(model, self.psi0.to(grid.dtype), t))
        self.U0 = model.local_potential(self.rho0, t)
        k2 = 0.0
        for i, ki in enumerate(grid.k64):
            k2 = k2 + (ki + (self.q if i == self.axis else 0.0)) ** 2
        self.kin = (0.5 * k2).to(grid.real_dtype)
        self.deflate = []

    def _fft(self, f):
        return torch.fft.fftn(f, dim=self.grid.dims)

    def _ifft(self, f):
        return torch.fft.ifftn(f, dim=self.grid.dims)

    def L(self, f):
        return self._ifft(self.kin * self._fft(f)) + (self.U0 - self.mu) * f

    def X(self, f):
        drho = self.psi0 * f
        dU = 0.0
        for term in self.model.local_terms:
            if getattr(term, "nonlinear", True):
                dU = dU + term.response(self.rho0, drho, self.q, self.axis)
        return self.psi0 * dU

    def M(self, f):
        return self.L(f) + 2 * self.X(f)

    def precondition(self, r, shift):
        return self._ifft(self._fft(r) / (self.kin + shift))

    def project(self, f):
        """Remove the components along the deflated fields (kernel of M, e.g. translation)."""
        for v in self.deflate:
            f = f - v * (v.conj() * f).sum(dim=self.grid.dims, keepdim=True)
        return f

    def solve_M(self, Y, shift, tol=1e-13, max_iter=5000):
        """M^-1 Y for a block Y (leading axis) by preconditioned conjugate gradients
        (within the complement of the deflated fields)."""
        dims = self.grid.dims
        dot = lambda a, b: (a.conj() * b).sum(dim=dims)
        expand = lambda c: c.reshape(c.shape + (1,) * self.grid.ndim)
        Y = self.project(Y)
        x = torch.zeros_like(Y)
        r = Y.clone()
        z = self.project(self.precondition(r, shift))
        p = z.clone()
        rz = dot(r, z)
        ynorm = torch.sqrt(dot(Y, Y).real)
        for _ in range(max_iter):
            Ap = self.M(p)
            alpha = rz / dot(p, Ap)
            x = x + expand(alpha) * p
            r = self.project(r - expand(alpha) * Ap)          # projected system P M P x = P Y
            if bool((torch.sqrt(dot(r, r).real) <= tol * ynorm).all()):
                return x
            z = self.project(self.precondition(r, shift))
            rz_new = dot(r, z)
            p = z + expand(rz_new / rz) * p
            rz = rz_new
        raise RuntimeError("bogoliubov: CG for M^-1 did not converge (is the state stable at this q?)")


def _inner(grid, a, b):
    """Gram matrix a^H b for blocks (m, *shape), (n, *shape) -> (m, n), times dV."""
    return torch.einsum("i...,j...->ij", a.conj(), b).reshape(a.shape[0], b.shape[0]) * grid.dV


def _rayleigh_ritz(grid, S, AS, BS, m):
    """Lowest m Ritz pairs of (S^H A S, S^H B S), dropping near-dependent directions."""
    GA, GB = _inner(grid, S, AS), _inner(grid, S, BS)
    GA, GB = 0.5 * (GA + GA.conj().T), 0.5 * (GB + GB.conj().T)
    try:
        wB, VB = torch.linalg.eigh(GB)
    except torch.linalg.LinAlgError:                     # nearly dependent block: regularize the Gram matrix
        eps = 1e-13 * float(GB.diagonal().real.abs().max())
        wB, VB = torch.linalg.eigh(GB + eps * torch.eye(GB.shape[0], dtype=GB.dtype, device=GB.device))
    keep = wB > 1e-12 * wB.max()
    T = VB[:, keep] / torch.sqrt(wB[keep])
    w, Y = torch.linalg.eigh(T.conj().T @ GA @ T)
    C = T @ Y
    return w[:m], C[:, :m]


def bogoliubov(model, psi0, q=0.0, n_modes=6, axis=-1, tol=1e-6, max_iter=1000, block=None, seed=0,
               shift=None, x0=None, deflate=None, verbose=False):
    """Lowest Bogoliubov frequencies at quasi-momentum q (along ``axis``).

    Returns (omega, modes, info): omega ascending (float64 tensor, sqrt of the
    eigenvalues clipped at 0), modes the corresponding f_- = u - v (Bloch-periodic
    part), info = {"iterations", "residual", "mu", "eigenvalues"}. ``tol`` bounds
    the relative residual ||L g - w^2 M^-1 g|| / (w^2 ||M^-1 g||) of every wanted
    mode (modes with w ~ 0, e.g. the gauge mode, on the scale of the largest
    wanted w^2). ``x0`` (block of fields) warm-starts the iteration; note it is
    in the g = M f_- representation, i.e. pass ``info["g"]`` of a nearby q.
    ``deflate`` (fields) removes zero modes of M from the problem: at q = 0 a
    modulated state has the translation mode d psi0/dz in the kernel of M, which
    makes M^-1 singular; pass [d psi0/dz] to compute the other modes (the
    translation Goldstone pair is then absent from the spectrum).
    """
    op = Bogoliubov(model, psi0, q, axis)
    grid = op.grid
    for v in deflate or []:                              # orthonormalize (plain sum inner product)
        v = op.project(v.to(grid.dtype))
        op.deflate.append(v / torch.sqrt((v.conj() * v).sum().real))
    m = block or n_modes + max(2, n_modes // 2)
    gen = torch.Generator().manual_seed(seed)
    shape = (m,) + grid.shape
    X = torch.complex(torch.randn(shape, generator=gen, dtype=torch.float64),
                      torch.randn(shape, generator=gen, dtype=torch.float64)).to(grid.device, grid.dtype)
    X = X * (op.psi0.abs() + 1e-3 * op.psi0.abs().max())
    if x0 is not None:                                   # warm start, e.g. modes of a nearby q
        k0 = min(len(x0), m)
        X[:k0] = x0[:k0].to(X.dtype)
    X = op.project(X)
    shift = shift if shift is not None else max(abs(op.mu), 1.0)
    apply_B = lambda Y: op.solve_M(Y, shift)
    apply_A = op.L
    BX = apply_B(X)
    AX = apply_A(X)
    lam, C = _rayleigh_ritz(grid, X, AX, BX, m)
    X, AX, BX = (torch.einsum("ij,i...->j...", C, Z) for Z in (X, AX, BX))
    P = AP = BP = None
    res = None
    for it in range(1, max_iter + 1):
        R = AX - lam.reshape(-1, *([1] * grid.ndim)) * BX
        rn = torch.sqrt(_inner(grid, R, R).diagonal().real)
        bn = torch.sqrt(_inner(grid, BX, BX).diagonal().real)
        scale = torch.clamp(lam.real.abs(), min=float(lam[:n_modes].real.abs().max()) * 1e-3 + 1e-300)
        res = rn / (bn * scale)                          # relative residual; ~0 modes measured on the block scale
        if verbose and it % 20 == 0:
            print(it, lam[:n_modes].tolist(), float(res[:n_modes].max()))
        if bool((res[:n_modes] <= tol).all()):
            break
        active = res > tol                               # soft locking: no new directions for converged vectors
        W = op.project(op.precondition(R[active], shift))
        BW = apply_B(W)
        AW = apply_A(W)
        blocks = [(X, AX, BX), (W, AW, BW)]
        if P is not None:
            blocks.append((P[active], AP[active], BP[active]))
        S = torch.cat([b[0] for b in blocks])
        AS = torch.cat([b[1] for b in blocks])
        BS = torch.cat([b[2] for b in blocks])
        d = _inner(grid, S, BS).diagonal().real
        keep = d > 1e-14 * float(d[:m].max())            # drop new directions with ~0 (or negative) B-norm
        keep[:m] = True
        if not bool(keep.all()):
            S, AS, BS, d = S[keep], AS[keep], BS[keep], d[keep]
        norms = torch.sqrt(d.clamp_min(1e-300))
        sc = (1 / norms).to(S.dtype).reshape(-1, *([1] * grid.ndim))
        S, AS, BS = S * sc, AS * sc, BS * sc
        try:
            lam_new, C = _rayleigh_ritz(grid, S, AS, BS, m)
        except torch.linalg.LinAlgError:
            if P is None:
                raise
            P = AP = BP = None                           # ill-conditioned with P: restart without it
            continue
        if float(lam_new.real.min()) < -1e-8 * float(lam_new.real.abs().max()) and P is not None:
            P = AP = BP = None                           # loss of positivity: restart without P
            continue
        lam = lam_new
        Xn = torch.einsum("ij,i...->j...", C, S)
        AXn = torch.einsum("ij,i...->j...", C, AS)
        BXn = torch.einsum("ij,i...->j...", C, BS)
        Cp = C.clone()
        Cp[:m] = 0                                       # the part of the update outside span(X)
        P = torch.einsum("ij,i...->j...", Cp, S)
        AP = torch.einsum("ij,i...->j...", Cp, AS)
        BP = torch.einsum("ij,i...->j...", Cp, BS)
        X, AX, BX = Xn, AXn, BXn
    omega = torch.sqrt(torch.clamp(lam[:n_modes].real, min=0.0)).to(torch.float64)
    return omega, BX[:n_modes], {"iterations": it, "residual": res[:n_modes].tolist(), "mu": op.mu,
                                "eigenvalues": lam[:n_modes].tolist(), "g": X[:n_modes]}

"""Validation of the quasi-1D tube model (dipgpe.quasi1d) against the literature and
against our full 3D eGPE results (docs/RESULTS.md, discontinuous tube transitions).

Literature (reduced theory with variational Gaussian widths), quoted in
benchmarks/tube_first_order.py: Blakie et al., arXiv:2004.12577,
n = 700/um: a* = 83.3, a_rot = 81.2; n = 6250/um: a* = 89.6, a_rot = 88.6.
Kirkby et al., PRR 7 (2025): l = 1.08 um, eta = 4.25 in the ground state at
88 a0, n = 2500/um; a_c = 91.05 a0 with these widths fixed.
Full 3D (ours): a_rot* = 84.50, 92.32, 89.83 and a* = 85.28 (85.32 for L -> inf),
92.32, 90.256 a0 for n = 700, 2500, 6250/um.

Steps
  uniform: variational widths of the uniform tube (analytic energy), and a_rot
           (roton instability of the uniform state with its own widths).
  ss88:    modulated ground state at 88 a0, n = 2500: widths and cell length
           minimizing E/N (1D ground states), to compare with Kirkby's.
Both kernels: "exact" (1D quadrature) and "closed" (Kirkby et al. Eq. (3)),
with the exact LHY Q5 (Kirkby et al.) or --lhy approx (Q5 ~ 1 + 3 eps^2/2,
Blakie et al. 2020).

  uv run python benchmarks/quasi1d_validation.py [uniform] [ss88]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dipgpe  # noqa: E402
from dipgpe.quasi1d import dipolar_kernel_1d, dipolar_kernel_1d_closed, transverse_energy, tube_model  # noqa: E402
from dipgpe.terms import lhy_coefficient as lhy_exact, q5  # noqa: E402
from dipgpe.units import OscillatorUnits  # noqa: E402

UNITS = OscillatorUnits(164, 2 * math.pi * 150)
A_DD = UNITS.bohr(130.8)
KERNELS = {"exact": dipolar_kernel_1d, "closed": dipolar_kernel_1d_closed}
LHY_MODE = ["exact"]


def lhy_coefficient(a_s, a_dd):
    """Exact Q5, or Blakie et al.'s Q5 ~ 1 + 3 eps^2 / 2 (--lhy approx)."""
    if LHY_MODE[0] == "exact":
        return lhy_exact(a_s, a_dd)
    eps = a_dd / a_s
    return lhy_exact(a_s, a_dd) / q5(eps) * (1 + 1.5 * eps * eps)


def nelder_mead(f, x0, step, tol=1e-10, max_iter=400):
    """Minimal Nelder-Mead in len(x0) dimensions; returns (x, f(x))."""
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step[j] if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (1 + abs(vals[0])):
            break
        c = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        r = [c[j] + (c[j] - pts[-1][j]) for j in range(n)]
        fr = f(r)
        if fr < vals[0]:
            e = [c[j] + 2 * (c[j] - pts[-1][j]) for j in range(n)]
            fe = f(e)
            pts[-1], vals[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = r, fr
        else:
            k = [c[j] + 0.5 * (pts[-1][j] - c[j]) for j in range(n)]
            fk = f(k)
            if fk < vals[-1]:
                pts[-1], vals[-1] = k, fk
            else:
                pts = [pts[0]] + [[pts[0][j] + 0.5 * (p[j] - pts[0][j]) for j in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i]


def uniform_energy(a_s, n, ell, eta, kernel):
    """E/N of the uniform tube in the reduced model (transverse energy included)."""
    g1 = 2 * a_s / ell ** 2
    gam1 = 2 * lhy_coefficient(a_s, A_DD) / (5 * math.pi ** 1.5 * ell ** 3)
    u0 = float(KERNELS[kernel](torch.tensor([0.0], dtype=torch.float64), 4 * math.pi * A_DD, ell, eta))
    return 0.5 * n * (g1 + u0) + 0.4 * gam1 * n ** 1.5 + transverse_energy(ell, eta)


def uniform_widths(a_s, n, kernel):
    (lg, le), E = nelder_mead(lambda p: uniform_energy(a_s, n, math.exp(p[0]), math.exp(p[1]), kernel),
                              [math.log(1.5), math.log(3.0)], [0.2, 0.2])
    return math.exp(lg), math.exp(le), E


def roton_gap2(a_s, n, ell, eta, kernel):
    g1 = 2 * a_s / ell ** 2
    gam1 = 2 * lhy_coefficient(a_s, A_DD) / (5 * math.pi ** 1.5 * ell ** 3)
    k = torch.linspace(1e-3, 8.0, 8000, dtype=torch.float64)
    e = 0.5 * k * k
    w2 = e * (e + 2 * n * (g1 + KERNELS[kernel](k, 4 * math.pi * A_DD, ell, eta) + 1.5 * gam1 * math.sqrt(n)))
    return float(w2.min())


def a_rot(n_um, kernel, lo=70.0, hi=100.0):
    n = UNITS.per_micron(n_um)
    f = lambda a: roton_gap2(UNITS.bohr(a), n, *uniform_widths(UNITS.bohr(a), n, kernel)[:2], kernel)
    assert f(lo) < 0 < f(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    a = 0.5 * (lo + hi)
    ell, eta, _ = uniform_widths(UNITS.bohr(a), n, kernel)
    return a, ell, eta


def modulated_energy(a_bohr, n_um, L, ell, eta, kernel, dx=0.05, state=None):
    """E/N (transverse included) of the modulated 1D ground state in one cell of length L."""
    grid = dipgpe.Grid(max(16, 2 * round(L / dx / 2)), L, dtype=torch.complex128, device="cuda")
    model = tube_model(grid, UNITS.bohr(a_bohr), A_DD, ell, eta, kernel=kernel)
    N = UNITS.per_micron(n_um) * L
    x = grid.x64[0]
    seed = state.get("psi") if state and state.get("psi") is not None and state["psi"].shape == grid.shape else \
        (1 + 0.5 * torch.cos(2 * math.pi * x / L)).to(torch.complex128)
    psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, seed, N), tol=1e-9)
    if state is not None:
        state["psi"] = psi
    line = psi.abs() ** 2
    contrast = float((line.max() - line.min()) / (line.max() + line.min()))
    return float(dipgpe.energy(model, psi)) / N + transverse_energy(ell, eta), contrast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("steps", nargs="*", default=["uniform", "ss88"])
    p.add_argument("--lhy", choices=["exact", "approx"], default="exact",
                   help="Q5 exact, or 1 + 3 eps^2/2 as in Blakie et al. 2020")
    args = p.parse_args()
    LHY_MODE[0] = args.lhy
    with dipgpe.runs.Run("quasi1d_validation", vars(args)) as run:
        um = UNITS.microns(1.0)
        if "uniform" in args.steps:
            for kernel in KERNELS:
                for n_um, lit in ((700, 81.2), (2500, None), (6250, 88.6)):
                    t0 = time.time()
                    a, ell, eta = a_rot(n_um, kernel)
                    print(f"[{kernel}] n = {n_um}: a_rot = {a:.3f} a0 (Blakie 2020: {lit}), widths at a_rot: "
                          f"l = {ell * um:.3f} um, eta = {eta:.3f} ({time.time() - t0:.1f} s)", flush=True)
                    run.result(**{f"a_rot_{kernel}_{n_um}": a, f"widths_{kernel}_{n_um}": [ell * um, eta]})
        if "ss88" in args.steps:
            for kernel in KERNELS:
                t0 = time.time()
                state = {}
                def f(pv):
                    L, ell, eta = math.exp(pv[0]), math.exp(pv[1]), math.exp(pv[2])
                    return modulated_energy(88.0, 2500, L, ell, eta, kernel, state=state)[0]
                (lL, lg, le), E = nelder_mead(f, [math.log(2.69 / um), math.log(1.08 / um), math.log(4.25)],
                                              [0.1, 0.1, 0.1], tol=1e-11)
                L, ell, eta = math.exp(lL), math.exp(lg), math.exp(le)
                _, c = modulated_energy(88.0, 2500, L, ell, eta, kernel, state=state)
                print(f"[{kernel}] 88 a0, n = 2500: l = {ell * um:.3f} um, eta = {eta:.3f}, cell {L * um:.3f} um, "
                      f"E/N = {E:.8f}, contrast {c:.3f} (Kirkby: l = 1.08 um, eta = 4.25) ({time.time() - t0:.0f} s)",
                      flush=True)
                run.result(**{f"ss88_{kernel}": {"l_um": ell * um, "eta": eta, "cell_um": L * um, "E_per_N": E,
                                                 "contrast": c}})


if __name__ == "__main__":
    main()

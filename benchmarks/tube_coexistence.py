"""Coexistence of the uniform and modulated tube states (Maxwell construction).

Steinberg, Maucher, Gurevich and Thiele, Phys. Rev. Research 7, L032044 (2025),
showed for the same 164Dy tube (omega_perp = 2 pi x 150 Hz, dipoles transverse)
that, because the atom number is conserved, a uniform superfluid at density n_U
and a modulated state at n_M can coexist when their chemical potentials and
pressures are equal. For mean densities between n_M and n_U the ground state of
a long tube is then phase separated (a "localized state": a modulated patch in
a uniform background), and a supercritical bifurcation at fixed density does
not by itself make the transition continuous.

Method. At fixed a_s and for each linear density n:
  uniform  z-uniform ground state -> e_U = E/N and mu_U;
  crystal  single-cell ground states at three cell lengths L, followed in n by
           continuation at fixed L (outward from --start in both directions,
           stopping where the state relaxes to the uniform tube); e_M(n) is the
           minimum of a parabola through e(L), and mu_M is interpolated to the
           optimal L (envelope theorem: there mu = df/dn).
With f = n e the energy per length, the one-dimensional pressure is
P = mu n - f. Coexistence is the crossing P_U(mu) = P_M(mu) of the two branches
in the (mu, P) plane (equivalently the double tangent of f(n)); the coexisting
densities follow from n = dP/dmu. Checks: Gibbs-Duhem (dP/dmu = n) along each
branch, and mu = df/dn from finite differences.
For a continuous onset the decisive quantity is the compressibility of the
modulated branch: d mu_M/dn < 0 near onset (equivalently, a common tangent)
means coexistence in a long tube. Near onset f_M - f_U = -(kappa/2)(n - n_c)^2,
so d mu_M/dn = d mu_U/dn - kappa: coexistence if kappa > d mu_U/dn.

Lengths in l = sqrt(hbar/m omega_perp), energies in hbar omega_perp; densities
are quoted in atoms per micron. All cells of one run use the same number of
grid points along z (--nz), so that e(L) is smooth in L.

Usage:
  uv run python benchmarks/tube_coexistence.py --a-list 90.15,90.256,90.35 \\
      --densities 6050,6450,25 --start 6150 --cells 4.2,4.3,4.4 --nz 17 --n-ref 6250
  uv run python benchmarks/tube_coexistence.py --a-list 92.25 --densities 2350,2650,10 \\
      --start 2500 --cells 4.3,4.45,4.6 --nz 18 --contrast-min 0.005 --n-ref 2500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_first_order import contrasts, tube_grid  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402

PER_L = UNITS.per_micron(1.0)          # atoms per l for one atom per micron


def values(model, psi, N, L):
    """Energy per atom, chemical potential and pressure of a stationary state."""
    E = float(dipgpe.energy(model, psi))
    mu = float(dipgpe.chemical_potential(model, psi))
    n = N / L
    return {"e": E / N, "mu": mu, "P": mu * n - E / L}


def uniform_branch(a_s, densities, dx, L_perp):
    grid = tube_grid(4.0, dx, L_perp, N_z=8)
    x, y, _ = grid.x64
    model = tube_model(grid, a_s)
    psi = torch.exp(-(x * x + y * y) / 2).expand(grid.shape).to(grid.dtype)
    rows = []
    for n in densities:
        N = n * PER_L * 4.0
        psi = dipgpe.normalize(grid, psi, N)
        t0 = time.time()
        psi, info = dipgpe.ground_state(model, psi)
        C_line, _, _ = contrasts(grid, psi)
        rows.append({"n": n, **values(model, psi, N, 4.0), "C_line": C_line,
                     "iterations": info["iterations"], "seconds": time.time() - t0})
    return rows


def crystal_sweep(model, grid, psi, densities, L, c_min, label, max_iter=200_000):
    """Continuation of the crystal along ``densities`` at fixed cell length L.
    Returns the rows and the converged state at the first density."""
    rows, first = [], None
    for n in densities:
        N = n * PER_L * L
        psi = dipgpe.normalize(grid, psi, N)
        t0 = time.time()
        psi, info = dipgpe.ground_state(model, psi, strict=False, max_iter=max_iter)
        C_line, C_axis, peaks = contrasts(grid, psi)
        row = {"n": n, "L": L, **values(model, psi, N, L), "C_line": C_line, "C_axis": C_axis,
               "droplets": peaks, "iterations": info["iterations"], "converged": bool(info["converged"]),
               "seconds": time.time() - t0}
        rows.append(row)
        if first is None:
            first = psi
        print(f"  {label} L = {L:.3f}  n = {n:7.1f}/um: e = {row['e']:.10f}  mu = {row['mu']:.10f}  "
              f"C = {C_line:.4f}  ({info['iterations']} it, {row['seconds']:.1f} s"
              f"{'' if row['converged'] else ', NOT CONVERGED'})", flush=True)
        if C_line < c_min or not row["converged"]:
            break
    return rows, first


def crystal_branch(a_s, densities, start, L, nz, dx, L_perp, width, c_min, max_iter=200_000):
    """Crystal states at fixed L for all densities reachable from ``start``."""
    grid = tube_grid(L, dx, L_perp, N_z=nz)
    x, y, z = grid.x64
    model = tube_model(grid, a_s)
    w = min(width, L / 6)
    seed = sum(torch.exp(-(z - m * L) ** 2 / (2 * w * w)) for m in (-1, 0, 1)) + 0.05
    psi0 = (torch.exp(-(x * x + y * y) / 2) * seed).to(grid.dtype)
    down = sorted((n for n in densities if n <= start), reverse=True)
    up = sorted(n for n in densities if n > start)
    rows, psi_start = crystal_sweep(model, grid, psi0, down, L, c_min, f"a = {a_s}", max_iter)
    if rows and rows[0]["C_line"] >= c_min and up:   # upward from the converged state at ``start``
        rows += crystal_sweep(model, grid, psi_start, up, L, c_min, f"a = {a_s}", max_iter)[0]
    return sorted(rows, key=lambda r: r["n"])


def optimal_crystal(rows, c_min):
    """Crystal minimized over L at each density from a parabola through e(L)."""
    by_n = {}
    for r in rows:
        if r["C_line"] >= c_min and r["converged"]:
            by_n.setdefault(r["n"], []).append(r)
    out = []
    for n, rs in sorted(by_n.items()):
        if len(rs) < 3:
            continue
        rs = sorted(rs, key=lambda r: r["L"])
        L = np.array([r["L"] for r in rs])
        e = np.array([r["e"] for r in rs])
        mu = np.array([r["mu"] for r in rs])
        c = np.polyfit(L - L.mean(), e, 2)
        if c[0] <= 0:
            continue
        Ls = L.mean() - c[1] / (2 * c[0])
        e_s = np.polyval(c, Ls - L.mean())
        mu_s = np.polyval(np.polyfit(L - L.mean(), mu, 2), Ls - L.mean())
        n_l = n * PER_L
        out.append({"n": n, "L_opt": float(Ls), "edge": bool(Ls < L.min() or Ls > L.max()),
                    "e": float(e_s), "mu": float(mu_s), "P": float(mu_s * n_l - n_l * e_s),
                    "C_line": float(np.interp(Ls, L, [r["C_line"] for r in rs]))})
    return out


def branch_checks(rows):
    """Gibbs-Duhem and mu = df/dn residuals along one branch (relative)."""
    if len(rows) < 3:
        return None
    n_l = np.array([r["n"] for r in rows]) * PER_L
    mu = np.array([r["mu"] for r in rows])
    f = n_l * np.array([r["e"] for r in rows])
    P = np.array([r["P"] for r in rows])
    dfdn = (f[2:] - f[:-2]) / (n_l[2:] - n_l[:-2])
    dPdmu = (P[2:] - P[:-2]) / (mu[2:] - mu[:-2])
    return {"mu_vs_dfdn_max_rel": float(np.max(np.abs(dfdn / mu[1:-1] - 1))),
            "dPdmu_vs_n_max_rel": float(np.max(np.abs(dPdmu / n_l[1:-1] - 1))),
            "dmu_dn": (np.diff(mu) / np.diff(n_l)).tolist()}


def coexistence(U, M):
    """Crossing of P_U(mu) and P_M(mu); coexisting densities from n = dP/dmu."""
    if len(U) < 3 or len(M) < 3:
        return None
    mu_U, P_U = np.array([r["mu"] for r in U]), np.array([r["P"] for r in U])
    mu_M, P_M = np.array([r["mu"] for r in M]), np.array([r["P"] for r in M])
    if np.any(np.diff(mu_M) <= 0) or np.any(np.diff(mu_U) <= 0):
        return {"crossing": None, "note": "mu not monotonic in n (negative compressibility)"}
    lo, hi = max(mu_U.min(), mu_M.min()), min(mu_U.max(), mu_M.max())
    if lo >= hi:
        return {"crossing": None, "note": "no overlap in mu"}
    m0, s = 0.5 * (lo + hi), 0.5 * (hi - lo)
    deg_U, deg_M = min(3, len(U) - 1), min(3, len(M) - 1)
    cU = np.polyfit((mu_U - m0) / s, P_U, deg_U)
    cM = np.polyfit((mu_M - m0) / s, P_M, deg_M)
    d = np.polysub(cM, cU)
    roots = [r.real for r in np.roots(d) if abs(r.imag) < 1e-9 and -1 <= r.real <= 1]
    dP = [float(np.polyval(d, -1)), float(np.polyval(d, 1))]
    if not roots:
        return {"crossing": None, "note": "no crossing inside the common mu range", "dP_at_ends": dP}
    x = roots[0]
    n_U = np.polyval(np.polyder(cU), x) / s / PER_L
    n_M = np.polyval(np.polyder(cM), x) / s / PER_L
    fit_U = float(np.max(np.abs(np.polyval(cU, (mu_U - m0) / s) - P_U)))
    fit_M = float(np.max(np.abs(np.polyval(cM, (mu_M - m0) / s) - P_M)))
    return {"crossing": True, "mu": float(m0 + s * x), "n_U": float(n_U), "n_M": float(n_M),
            "dn": float(n_U - n_M), "fit_resid_P": [fit_U, fit_M], "dP_at_ends": dP}


def onset(U, crystal_rows, c_min):
    """Onset analysis at fixed L: kappa from f_M - f_U = -(kappa/2)(n - n_c)^2
    (fit over the crystal states) against the uniform d mu/dn."""
    out = {}
    uni = {r["n"]: r for r in U}
    for L in sorted({r["L"] for r in crystal_rows}):
        rs = [r for r in crystal_rows if r["L"] == L and r["C_line"] >= c_min and r["converged"] and r["n"] in uni]
        if len(rs) < 4:
            continue
        n = np.array([r["n"] for r in rs])
        n_l = n * PER_L
        df = n_l * np.array([r["e"] - uni[r["n"]]["e"] for r in rs])
        mu_M = np.array([r["mu"] for r in rs])
        mu_U = np.array([uni[r["n"]]["mu"] for r in rs])
        C = np.array([r["C_line"] for r in rs])
        out[str(L)] = {"n": n.tolist(), "C_line": C.tolist(), "df": df.tolist(),
                       "dmu_dn_M": (np.diff(mu_M) / np.diff(n_l)).tolist(),
                       "dmu_dn_U": (np.diff(mu_U) / np.diff(n_l)).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-list", type=str, required=True, help="comma-separated scattering lengths (a0)")
    ap.add_argument("--densities", type=str, required=True, help="start,stop,step in atoms per micron")
    ap.add_argument("--start", type=float, required=True, help="density where every crystal branch exists")
    ap.add_argument("--cells", type=str, required=True, help="three or more cell lengths (l)")
    ap.add_argument("--nz", type=int, required=True, help="grid points per cell along z (same for all L)")
    ap.add_argument("--dx", type=float, default=0.25)
    ap.add_argument("--L-perp", type=float, default=24.0)
    ap.add_argument("--width", type=float, default=1.0, help="droplet width of the crystal seed (l)")
    ap.add_argument("--contrast-min", type=float, default=0.05, help="line contrast below which a state is uniform")
    ap.add_argument("--n-ref", type=float, default=None, help="density for the coexistence window in a_s")
    ap.add_argument("--max-iter", type=int, default=200_000, help="iteration cap per crystal state (unconverged states end a sweep)")
    args = ap.parse_args()
    with dipgpe.runs.Run("tube_coexistence", vars(args)) as run:
        return _run(args, run)


def _run(args, run):
    a_list = [float(v) for v in args.a_list.split(",")]
    n0, n1, dn = [float(v) for v in args.densities.split(",")]
    densities = [round(n0 + i * dn, 3) for i in range(int(round((n1 - n0) / dn)) + 1)]
    cells = [float(v) for v in args.cells.split(",")]
    t_start = time.time()
    per_a = []
    for a in a_list:
        t_a = time.time()
        U = uniform_branch(a, densities, args.dx, args.L_perp)
        print(f"a = {a}: uniform states ({time.time() - t_a:.0f} s), max line contrast "
              f"{max(r['C_line'] for r in U):.1e}", flush=True)
        rows = []
        for L in cells:
            rows += crystal_branch(a, densities, args.start, L, args.nz, args.dx, args.L_perp, args.width,
                                   args.contrast_min, args.max_iter)
        M = optimal_crystal(rows, args.contrast_min)
        entry = {"a_s": a, "uniform": U, "crystal_rows": rows, "crystal_opt": M,
                 "checks": {"uniform": branch_checks(U), "crystal": branch_checks(M)},
                 "coexistence": coexistence(U, M), "onset": onset(U, rows, args.contrast_min),
                 "seconds": time.time() - t_a}
        if args.n_ref is not None:
            uni = {r["n"]: r for r in U}
            ref = [m for m in M if m["n"] == args.n_ref and args.n_ref in uni]
            entry["de_at_n_ref"] = ref[0]["e"] - uni[args.n_ref]["e"] if ref else None
        per_a.append(entry)
        co = entry["coexistence"]
        print(f"a = {a}: {len(M)} optimized crystal densities, checks {entry['checks']['uniform'] and {k: v for k, v in entry['checks']['uniform'].items() if k != 'dmu_dn'}}; "
              f"coexistence {co if co is None or not co.get('crossing') else {k: co[k] for k in ('mu', 'n_M', 'n_U', 'dn')}} "
              f"({time.time() - t_a:.0f} s)", flush=True)
    result = {"a_list": a_list, "densities": densities, "cells": cells, "nz": args.nz, "dx": args.dx,
              "L_perp": args.L_perp, "per_a": per_a, "seconds": time.time() - t_start}
    co = [(e["a_s"], e["coexistence"]) for e in per_a if e["coexistence"] and e["coexistence"].get("crossing")]
    if args.n_ref is not None and len(co) >= 2:
        a = np.array([c[0] for c in co])
        nM = np.array([c[1]["n_M"] for c in co])
        nU = np.array([c[1]["n_U"] for c in co])
        pM, pU = np.polyfit(a, nM, 1), np.polyfit(a, nU, 1)
        a_lo, a_hi = (args.n_ref - pM[1]) / pM[0], (args.n_ref - pU[1]) / pU[0]
        result["window"] = {"n_ref": args.n_ref, "a_from_crystal_side": float(min(a_lo, a_hi)),
                            "a_from_uniform_side": float(max(a_lo, a_hi)), "width": float(abs(a_hi - a_lo)),
                            "dn_coex_mean": float(np.mean(nU - nM)), "slope_n_per_a0": float(np.mean([pM[0], pU[0]]))}
        de = [(e["a_s"], e["de_at_n_ref"]) for e in per_a if e.get("de_at_n_ref") is not None]
        if len(de) >= 2:   # e_M - e_U is curved in a_s: quadratic through three or more points
            a_de, v_de = np.array([d[0] for d in de]), np.array([d[1] for d in de])
            p = np.polyfit(a_de - a_de.mean(), v_de, min(2, len(de) - 1))
            roots = [r.real + a_de.mean() for r in np.roots(p) if abs(r.imag) < 1e-12
                     and a_de.min() <= r.real + a_de.mean() <= a_de.max()]
            result["window"]["a_star_uniform_density"] = float(roots[0]) if roots else None
        print(f"\ncoexistence window at n = {args.n_ref:.0f}/um: {result['window']}", flush=True)
    output = run.save_json("tube_coexistence.json", result)
    run.result(**{"window": result.get("window"),
                  "coexistence": {str(e["a_s"]): e["coexistence"] for e in per_a}})
    print(f"wrote {output} ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()

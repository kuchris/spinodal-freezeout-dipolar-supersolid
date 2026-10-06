"""Discontinuous vs continuous superfluid-crystal transition of the 164Dy tube.

Literature, quoted verbatim (Blakie et al., arXiv:2004.12577, reduced theory):
n = 0.7×10³/µm: "a transition point at a* = 83.3a0, identified as where the
energy per particle E of the modulated stationary solution crosses EBEC ... This
transition point is appreciably higher than arot = 81.2a0, where the BEC becomes
dynamically unstable, meaning that hysteresis can occur"; the unit cell "is
larger than (and disconnected from) the roton wavelength". n = 6.25×10³/µm:
"the transition occurs at a* = 89.6a0 (cf. arot = 88.6a0)". At intermediate
density (2.5×10³/µm) the transition is continuous, a* = a_rot*.

Method. For each cell length L the crystal branch is followed upward in a_s by
continuation: each ground state (dipgpe.ground_state, conjugate gradient) starts
from the converged crystal at the previous a_s, so it stays on the branch while
the branch exists, also where it is only metastable. The branch has ended when
the state relaxes to the uniform tube (line contrast < 0.05). The uniform tube
energy comes from a z-uniform ground state. Per L:
  a*(L)     where dE = E_crystal/N - E_uniform/N changes sign (linear interpolation),
  a_end(L)  between the last a_s with a crystal and the first without,
  C(a*)     line contrast of the crystal at a*.
a* = max over L of a*(L). First order: a* > a_rot* beyond the numerical
uncertainty, C(a*) finite, and a crystal branch with dE > 0 above a* (hysteresis).
Continuous (control, n = 2500): a* = a_rot*, C(a*) -> 0, no branch above a*.
a_rot* comes from tube_supersolid.py (roton softening).

Usage:
  uv run python benchmarks/tube_first_order.py --density 700 [--dx 0.25] [--output ...]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402

SETUPS = {  # density: a_rot* (tube_supersolid.py), a_s scan, cells, transverse box, seed width
    700.0: {"a_rot": 84.50, "a": (84.25, 85.6, 0.05), "cells": [8, 12, 16, 20, 24, 32, 40, 48, 64],
            "L_perp": 16.0, "width": 0.6},
    2500.0: {"a_rot": 92.32, "a": (91.75, 92.6, 0.05), "cells": [4.2, 4.35, 4.5, 4.65, 4.8, 5.5, 6.5],
             "L_perp": 24.0, "width": 1.0},
    6250.0: {"a_rot": 89.83, "a": (89.25, 91.0, 0.05), "cells": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0],
             "L_perp": 24.0, "width": 1.0},
}


def tube_grid(L, dx, L_perp, N_z=None):
    Np = int(round(L_perp / dx))
    return dipgpe.Grid((Np, Np, N_z or max(8, int(round(L / dx)))), (L_perp, L_perp, L),
                   dtype=torch.complex128, device="cuda")


def contrasts(grid, psi):
    rho = psi.abs() ** 2
    line, axis = rho.sum((0, 1)), rho[grid.shape[0] // 2, grid.shape[1] // 2, :]
    c = lambda f: float((f.max() - f.min()) / (f.max() + f.min()))
    mid = 0.5 * (line.max() + line.min())
    peaks = int(((line > torch.roll(line, 1)) & (line >= torch.roll(line, -1)) & (line > mid)).sum())
    return c(line), c(axis), peaks


def uniform_energy(n, a_s, dx, L_perp):
    grid = tube_grid(4.0, dx, L_perp, N_z=8)
    x, y, _ = grid.x64
    N = UNITS.per_micron(n) * 4.0
    model = tube_model(grid, a_s)
    psi0 = dipgpe.normalize(grid, torch.exp(-(x * x + y * y) / 2).expand(grid.shape).to(grid.dtype), N)
    psi, _ = dipgpe.ground_state(model, psi0)
    return float(dipgpe.energy(model, psi)) / N


def crystal_branch(n, L, a_list, E_uniform, dx, L_perp, width):
    """Continuation of the crystal branch upward in a_s at cell length L."""
    grid = tube_grid(L, dx, L_perp)
    x, y, z = grid.x64
    N = UNITS.per_micron(n) * L
    w = min(width, L / 6)
    f = sum(torch.exp(-(z - m * L) ** 2 / (2 * w * w)) for m in (-1, 0, 1)) + 0.05
    psi = dipgpe.normalize(grid, (torch.exp(-(x * x + y * y) / 2) * f).to(grid.dtype), N)
    rows = []
    for a in a_list:
        model = tube_model(grid, a)
        t0 = time.time()
        psi, info = dipgpe.ground_state(model, psi)
        E = float(dipgpe.energy(model, psi)) / N
        C_line, C_axis, peaks = contrasts(grid, psi)
        crystal = C_line >= 0.05
        rows.append({"a_s": a, "dE": E - E_uniform[a], "C_line": C_line, "C_axis": C_axis,
                     "droplets": peaks if crystal else 0, "iterations": info["iterations"],
                     "seconds": time.time() - t0})
        print(f"  L = {L:5.2f}  a_s = {a:6.3f}: dE = {E - E_uniform[a]:+.4e}  C = {C_line:.3f} "
              f"(axis {C_axis:.3f})  droplets {peaks if crystal else 0}  "
              f"({info['iterations']} it, {time.time() - t0:.1f} s)", flush=True)
        if not crystal:
            break
    return rows


def analyse(rows):
    """a*(L), contrast at a*, branch end and transition type, from one continuation.

    Crossings are taken only between consecutive crystal states with the same
    number of droplets per cell (a branch that splits or merges droplets is a
    different branch). If dE stays negative up to the branch end, the branch
    joins the uniform state continuously and a* is the end of the branch."""
    out = {"a_star": None, "C_at_a_star": None, "a_end": None, "metastable_dE_max": None, "kind": None}
    crystal = [r for r in rows if r["C_line"] >= 0.05]
    for r0, r1 in zip(crystal, crystal[1:]):
        if r0["droplets"] == r1["droplets"] and r0["dE"] < 0 <= r1["dE"]:
            s = -r0["dE"] / (r1["dE"] - r0["dE"])
            out.update(a_star=r0["a_s"] + s * (r1["a_s"] - r0["a_s"]),
                       C_at_a_star=r0["C_line"] + s * (r1["C_line"] - r0["C_line"]),
                       droplets=r0["droplets"], kind="crossing")
    if len(crystal) < len(rows):
        out["a_end"] = [crystal[-1]["a_s"] if crystal else None, rows[len(crystal)]["a_s"]]
    above = [r["dE"] for r in crystal if r["dE"] > 0]
    out["metastable_dE_max"] = max(above) if above else None
    if out["a_star"] is None and crystal and out["a_end"] and all(r["dE"] < 0 for r in crystal):
        last = crystal[-1]
        out.update(a_star=0.5 * (out["a_end"][0] + out["a_end"][1]), C_at_a_star=last["C_line"],
                   droplets=last["droplets"], kind="continuous", dE_last=last["dE"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=float, required=True, choices=sorted(SETUPS))
    ap.add_argument("--dx", type=float, default=0.25)
    ap.add_argument("--cells", type=str, default=None, help="comma-separated cell lengths (l)")
    ap.add_argument("--a-range", type=str, default=None, help="start,stop,step in a0")
    ap.add_argument("--L-perp", type=float, default=None, help="transverse box (l); default per density")
    ap.add_argument("--output", type=Path, default=None, help="extra copy of the result JSON (the run directory always keeps one)")
    args = ap.parse_args()
    with dipgpe.runs.Run("tube_first_order", vars(args)) as run:
        return _run(args, run)


def _run(args, run):
    setup = dict(SETUPS[args.density])
    if args.L_perp:
        setup["L_perp"] = args.L_perp
    a0, a1, da = [float(v) for v in args.a_range.split(",")] if args.a_range else setup["a"]
    a_list = [round(a0 + i * da, 4) for i in range(int(round((a1 - a0) / da)) + 1)]
    cells = [float(v) for v in args.cells.split(",")] if args.cells else setup["cells"]
    t_start = time.time()
    E_uniform = {a: uniform_energy(args.density, a, args.dx, setup["L_perp"]) for a in a_list}
    print(f"n = {args.density:.0f}/um, dx = {args.dx}, L_perp = {setup['L_perp']}, a_rot* = {setup['a_rot']} a0; "
          f"uniform states {time.time() - t_start:.0f} s", flush=True)
    per_cell = []
    for L in cells:
        rows = crystal_branch(args.density, L, a_list, E_uniform, args.dx, setup["L_perp"], setup["width"])
        per_cell.append({"L": L, "L_um": UNITS.microns(L), **analyse(rows), "rows": rows})
        c = per_cell[-1]
        print(f"L = {L} ({UNITS.microns(L):.2f} um): {c['kind']} a* = {c['a_star']}, C(a*) = {c['C_at_a_star']}, "
              f"droplets/cell {c.get('droplets')}, branch end in {c['a_end']}", flush=True)
    found = [c for c in per_cell if c["a_star"] is not None]
    best = max(found, key=lambda c: c["a_star"]) if found else None
    result = {"density": args.density, "dx": args.dx, "L_perp": setup["L_perp"], "a_rot": setup["a_rot"],
              "a_list": a_list, "cells": per_cell, "seconds": time.time() - t_start}
    if best:
        result.update(a_star=best["a_star"], L_star=best["L"], C_at_a_star=best["C_at_a_star"],
                      a_end=best["a_end"], kind=best["kind"], droplets=best.get("droplets"))
        print(f"\nn = {args.density:.0f}/um ({best['kind']}, {best.get('droplets')} droplet(s) per cell): "
              f"a* = {best['a_star']:.3f} a0 at L = {best['L']} l "
              f"({UNITS.microns(best['L']):.2f} um), a* - a_rot* = {best['a_star'] - setup['a_rot']:+.3f} a0, "
              f"C(a*) = {best['C_at_a_star']:.3f}, crystal branch ends in {best['a_end']} a0 "
              f"({time.time() - t_start:.0f} s)", flush=True)
    output = run.save_json("tube_first_order.json", result, also=args.output)
    run.result(**{k: result.get(k) for k in ("a_star", "L_star", "C_at_a_star", "a_end", "kind")})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

"""Localized states (a crystal patch in a uniform background) in a long periodic tube.

Inside the coexistence window found by tube_coexistence.py, a long tube at mean
density n_bar can lower its bulk energy by phase separation into a crystal at
n_M and a uniform superfluid at n_U (Steinberg, Maucher, Gurevich and Thiele,
Phys. Rev. Research 7, L032044 (2025)). In a finite tube the two interfaces
cost energy, so whether the resulting localized state is the ground state
depends on the tube length. This script constructs the localized state directly
in the full 3D model and compares its energy with the homogeneous states on the
same grid.

Method.
  1. Coexisting densities n_M, n_U at the cell length of the tube (fixed L,
     same grid spacing along z), from the Maxwell construction of
     tube_coexistence.py over n_bar +- 150/um.
  2. Single-cell ground states: crystal at n_M, uniform at n_U (the seed), and
     crystal and uniform at n_bar (the homogeneous references; the tiled cell is
     an exact stationary state of the tube, with the same energy per atom).
  3. Seed: K cells of the crystal followed by cells - K cells of the uniform
     state, renormalized to N = n_bar x length, then minimized at fixed N
     (dipgpe.ground_state, conjugate gradient). Several K test whether the
     patch size is unique.
  4. Optional stability test: perturb the lowest localized state with complex
     noise of relative amplitude --perturb and minimize again; a local minimum
     returns to the same energy and patch.
  Bulk coexistence energy (no interfaces, lever rule):
     e_coex = [x n_M e_M(n_M) + (1 - x) n_U e_U(n_U)] / n_bar, x = (n_U - n_bar)/(n_U - n_M).
  Interface energy: sigma = (E_loc - N e_coex) / 2 (two interfaces in the periodic
  tube). Gain per length of phase separation over the best homogeneous state:
  g = n_bar [min(e_U, e_M)(n_bar) - e_coex]; the localized state can only be the
  ground state for tube lengths above about L_min = 2 sigma / g.
  Patch and background densities are averages of the one-cell running mean of
  the line density, at least two cells away from the interfaces.

Lengths in l, energies in hbar omega_perp; densities quoted in atoms per micron.

Usage:
  uv run python benchmarks/tube_localized.py --a 90.25 --cells 32 --patch 16,21,26 --perturb 1e-3
  uv run python benchmarks/tube_localized.py --a 90.25 --cells 64
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_coexistence import PER_L, coexistence, crystal_branch, uniform_branch  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402


def cell_state(a_s, grid, n, crystal):
    """Single-cell ground state (crystal seeded with one droplet at z = 0, or uniform)."""
    x, y, z = grid.x64
    L = grid.length[2]
    f = (1 + 0.5 * torch.cos(2 * math.pi * z / L)) if crystal else torch.ones_like(z)
    seed = (torch.exp(-(x * x + y * y) / 2) * f).expand(grid.shape).to(torch.complex128)
    model = tube_model(grid, a_s)
    N = n * PER_L * L
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, seed, N), preconditioner="combined")
    line = (psi.abs() ** 2).sum((0, 1))
    return psi.abs().to(torch.complex128), {
        "n": n, "e": float(dipgpe.energy(model, psi)) / N, "mu": float(dipgpe.chemical_potential(model, psi)),
        "C_line": float((line.max() - line.min()) / (line.max() + line.min())), "iterations": info["iterations"]}


def coexistence_at_cell(a_s, n_bar, L, nz, dx, L_perp, span=150.0, step=25.0):
    densities = [round(n_bar - span + i * step, 3) for i in range(int(round(2 * span / step)) + 1)]
    U = uniform_branch(a_s, densities, dx, L_perp)
    rows = crystal_branch(a_s, densities, densities[0], L, nz, dx, L_perp, 1.0, 0.05)
    M = [r for r in rows if r["C_line"] >= 0.05 and r["converged"]]
    return coexistence(U, M)


def analyse(grid, model, psi, N, nz_cell, L_cell, shift):
    """Energy, residual and patch geometry; the line density is rolled by ``shift``
    so that the background centre sits at the array ends."""
    line = np.roll(((psi.abs() ** 2).sum((0, 1)) * grid.dx[0] * grid.dx[1]).double().cpu().numpy(), shift)
    h = nz_cell // 2
    win = sliding_window_view(np.concatenate([line[-h:], line, line[:h]]), 2 * h + 1)
    mx, mn, mean = win.max(1), win.min(1), win.mean(1)
    contrast = (mx - mn) / (mx + mn)
    patch, back = contrast > 0.3, contrast < 0.05
    m = 2 * nz_cell                               # two cells away from any interface

    def core(mask):
        ext = np.concatenate([mask[-m:], mask, mask[:m]])
        return sliding_window_view(ext, 2 * m + 1).all(1)
    cp, cb = core(patch), core(back)
    dz = grid.dx[2]
    rise = []
    left_back = np.where(back[: line.size // 2])[0]
    left_patch = np.where(contrast[: line.size // 2] > 0.6)[0]
    right_back = np.where(back[line.size // 2:])[0] + line.size // 2
    right_patch = np.where(contrast[line.size // 2:] > 0.6)[0] + line.size // 2
    if len(left_back) and len(left_patch) and len(right_back) and len(right_patch):
        rise = [(left_patch[0] - left_back[-1]) * dz, (right_back[0] - right_patch[-1]) * dz]
    avg = line.mean()
    peaks = lambda thr: int(((line > np.roll(line, 1)) & (line >= np.roll(line, -1)) & (line > thr * avg)).sum())
    return {
        "e": float(dipgpe.energy(model, psi)) / N, "mu": float(dipgpe.chemical_potential(model, psi)),
        "residual": float(dipgpe.residual(model, psi)),
        "modulated_cells": float(patch.sum() * dz / L_cell), "droplets_1.2": peaks(1.2), "droplets_1.5": peaks(1.5),
        "n_patch": float(mean[cp].mean() / PER_L) if cp.any() else None,
        "n_background": float(mean[cb].mean() / PER_L) if cb.any() else None,
        "interface_width_um": [UNITS.microns(r) for r in rise] if rise else None,
        "line": line.tolist()}


def relax(model, grid, psi, label, max_iter, N):
    t0 = time.time()

    def progress(it, f):
        print(f"    {label}: it {it:6d}  {time.time() - t0:6.1f} s  E/N = {float(dipgpe.energy(model, f)) / N:.12f}",
              flush=True)
        return False
    psi, info = dipgpe.ground_state(model, psi, strict=False, max_iter=max_iter, preconditioner="combined",
                                    callback=progress, callback_every=500)
    return psi, info, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=float, required=True, help="scattering length (a0)")
    ap.add_argument("--density", type=float, default=6250.0, help="mean density (atoms per micron)")
    ap.add_argument("--cells", type=int, default=32, help="tube length in crystal cells")
    ap.add_argument("--cell-um", type=float, default=2.76)
    ap.add_argument("--nz-cell", type=int, default=17, help="grid points per cell along z")
    ap.add_argument("--patch", type=str, default="auto", help="comma-separated initial crystal cells, or auto (lever rule)")
    ap.add_argument("--perturb", type=float, default=0.0, help="relative noise for the stability test (0: skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dx", type=float, default=0.25)
    ap.add_argument("--L-perp", type=float, default=24.0)
    ap.add_argument("--max-iter", type=int, default=60_000)
    args = ap.parse_args()
    with dipgpe.runs.Run("tube_localized", vars(args)) as run:
        return _run(args, run)


def _run(args, run):
    t_start = time.time()
    L = args.cell_um / UNITS.microns(1.0)
    Np = int(round(args.L_perp / args.dx))
    cg = dipgpe.Grid((Np, Np, args.nz_cell), (args.L_perp, args.L_perp, L), dtype=torch.complex128, device="cuda")
    co = coexistence_at_cell(args.a, args.density, L, args.nz_cell, args.dx, args.L_perp)
    if not co or not co.get("crossing"):
        raise RuntimeError(f"no coexistence at a = {args.a}: {co}")
    n_M, n_U = co["n_M"], co["n_U"]
    if not n_M < args.density < n_U:
        raise RuntimeError(f"n_bar = {args.density} outside the coexistence interval {n_M:.1f}-{n_U:.1f}")
    psi_M, s_M = cell_state(args.a, cg, n_M, True)
    psi_U, s_U = cell_state(args.a, cg, n_U, False)
    _, h_M = cell_state(args.a, cg, args.density, True)
    _, h_U = cell_state(args.a, cg, args.density, False)
    x = (n_U - args.density) / (n_U - n_M)
    e_coex = (x * n_M * s_M["e"] + (1 - x) * n_U * s_U["e"]) / args.density
    print(f"a = {args.a} a0: coexistence n_M = {n_M:.2f}, n_U = {n_U:.2f}/um, lever-rule crystal fraction {x:.3f}; "
          f"e_U(n_bar) - e_M(n_bar) = {h_U['e'] - h_M['e']:+.3e}, e_coex - e_M(n_bar) = {e_coex - h_M['e']:+.3e} "
          f"({time.time() - t_start:.0f} s)", flush=True)

    grid = dipgpe.Grid((Np, Np, args.cells * args.nz_cell), (args.L_perp, args.L_perp, args.cells * L),
                       dtype=torch.complex128, device="cuda")
    model = tube_model(grid, args.a)
    N = args.density * PER_L * args.cells * L
    patches = [round(x * args.cells)] if args.patch == "auto" else [int(v) for v in args.patch.split(",")]
    states, best = [], None
    for K in patches:
        psi = torch.cat([psi_M.repeat(1, 1, K), psi_U.repeat(1, 1, args.cells - K)], dim=2).contiguous()
        psi, info, sec = relax(model, grid, dipgpe.normalize(grid, psi, N), f"K = {K}", args.max_iter, N)
        shift = -(K * args.nz_cell + (args.cells - K) * args.nz_cell // 2)
        st = {"K": K, "iterations": info["iterations"], "converged": bool(info["converged"]), "seconds": sec,
              **analyse(grid, model, psi, N, args.nz_cell, L, shift)}
        states.append(st)
        print(f"  K = {K}: E/N - e_M(n_bar) = {st['e'] - h_M['e']:+.4e}, residual {st['residual']:.1e}, "
              f"modulated {st['modulated_cells']:.2f} cells, droplets {st['droplets_1.2']} (1.2x) "
              f"{st['droplets_1.5']} (1.5x), n_patch {st['n_patch']}, n_background {st['n_background']}, "
              f"interfaces {st['interface_width_um']} um ({info['iterations']} it, {sec:.0f} s)", flush=True)
        if best is None or st["e"] < best[0]["e"]:
            best = (st, psi, shift)
    loc, psi_loc, shift = best
    sigma = (loc["e"] - e_coex) * N / 2
    gain = args.density * PER_L * (min(h_U["e"], h_M["e"]) - e_coex)
    result = {"a_s": args.a, "density": args.density, "cells": args.cells, "tube_um": args.cells * args.cell_um,
              "coexistence": co, "n_M": n_M, "n_U": n_U, "lever_fraction": x,
              "cell_states": {"crystal_nM": s_M, "uniform_nU": s_U, "crystal_nbar": h_M, "uniform_nbar": h_U},
              "e_coex": e_coex, "states": states,
              "localized_minus_crystal": loc["e"] - h_M["e"], "localized_minus_uniform": loc["e"] - h_U["e"],
              "sigma_hbar_omega": sigma, "gain_per_l": gain, "gain_tube": gain * args.cells * L,
              "L_min_um": UNITS.microns(2 * sigma / gain) if gain > 0 else None}
    if args.perturb > 0:
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        noise = torch.complex(torch.randn(grid.shape, generator=g, device="cuda", dtype=torch.float64),
                              torch.randn(grid.shape, generator=g, device="cuda", dtype=torch.float64))
        psi_p = dipgpe.normalize(grid, psi_loc * (1 + args.perturb * noise), N)
        e_p = float(dipgpe.energy(model, psi_p)) / N
        psi_p, info, sec = relax(model, grid, psi_p, "perturbed", args.max_iter, N)
        st = analyse(grid, model, psi_p, N, args.nz_cell, L, shift)
        result["perturbed"] = {"amplitude": args.perturb, "e_after_noise": e_p, "e_relaxed": st["e"],
                               "de_relaxed": st["e"] - loc["e"], "modulated_cells": st["modulated_cells"],
                               "residual": st["residual"], "iterations": info["iterations"], "seconds": sec}
        print(f"  perturbed by {args.perturb}: E/N rose by {e_p - loc['e']:+.2e}, relaxed back to "
              f"{st['e'] - loc['e']:+.2e} relative to the localized state, modulated {st['modulated_cells']:.2f} cells "
              f"({info['iterations']} it, {sec:.0f} s)", flush=True)
    col = np.roll((psi_loc.abs() ** 2).sum(0).double().cpu().numpy() * grid.dx[0], shift, axis=1).astype(np.float32)
    np.savez_compressed(run.file("localized_column.npz"), column_yz=col, dz_um=UNITS.microns(grid.dx[2]),
                        dy_um=UNITS.microns(grid.dx[1]))
    output = run.save_json("tube_localized.json", result)
    run.result(**{k: result[k] for k in ("a_s", "cells", "n_M", "n_U", "lever_fraction", "localized_minus_crystal",
                                         "localized_minus_uniform", "sigma_hbar_omega", "gain_tube", "L_min_um")})
    print(f"\nlocalized - crystal = {result['localized_minus_crystal']:+.4e}, localized - uniform = "
          f"{result['localized_minus_uniform']:+.4e} per atom; sigma = {sigma:.2f} hbar omega per interface; "
          f"bulk gain over the tube {result['gain_tube']:.2f} hbar omega; L_min = {result['L_min_um']} um")
    print(f"wrote {output} ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()

"""Bogoliubov spectrum of the tube crystal near the end of its branch (paper C1).

The crystal cell of the 3D ramps (6250/um, cell 2.76 um, dx = 0.25 l, L_perp =
24 l, the model of tube_ramp.py) is continued upwards in a_s from a warm start,
and the lowest Bogoliubov frequencies are computed at quasi-momenta q = f pi/L_cell
(f = 0: same period as the cell; f = 1: zone edge). The gauge and translation
modes are zero at q = 0; the translation mode d psi/dz is a zero mode of M and is
deflated (removed), so at q = 0 the spectrum starts with the gauge mode and the
lowest other mode is the candidate soft mode of the branch end. For a saddle-node (fold) end, omega^2 vanishes as
sqrt(a_end - a_s), i.e. omega^4 is linear in a_s; the zero of the fitted line
estimates a_end. The continuation stops when the crystal is lost (line
contrast < 0.05).

  uv run python benchmarks/tube_crystal_bdg.py [--a-list 90.0,90.2,90.3,90.35,90.38]
      [--q-fracs 0,1] [--n-modes 4] [--tol 1e-4]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from dipgpe.bdg import bogoliubov  # noqa: E402
from tube_ramp import A_DD, UNITS  # noqa: E402


def cell_model(grid, a_s):
    x, y, _ = grid.x64
    a = UNITS.bohr(a_s)
    return dipgpe.GPE(grid, dipgpe.Potential((0.5 * (x * x + y * y)).expand(grid.shape).to(grid.real_dtype)),
                  dipgpe.Contact(4 * math.pi * a),
                  dipgpe.Dipolar(grid, 4 * math.pi * A_DD, direction=(0, 1, 0), cutoff="tube"),
                  dipgpe.LHY(dipgpe.lhy_coefficient(a, A_DD)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-list", default="90.0,90.2,90.3,90.35,90.38")
    p.add_argument("--q-fracs", default="0,1")
    p.add_argument("--n-modes", type=int, default=4)
    p.add_argument("--density", type=float, default=6250.0)
    p.add_argument("--cell-um", type=float, default=2.76)
    p.add_argument("--cells", type=int, default=32, help="cells of the ramp tube (sets nz as in tube_ramp.py)")
    p.add_argument("--dx", type=float, default=0.25)
    p.add_argument("--L-perp", type=float, default=24.0)
    p.add_argument("--tol", type=float, default=1e-4, help="relative residual (frequencies to ~tol^2)")
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--warm", action="store_true",
                   help="warm-start each a_s from the previous modes (slower here and can be ill-conditioned)")
    args = p.parse_args()
    um = UNITS.microns(1.0)
    L = args.cells * args.cell_um / um
    nz = round(L / args.cells / args.dx)
    Np = 2 * round(args.L_perp / args.dx / 2)
    n = UNITS.per_micron(args.density)
    grid = dipgpe.Grid((Np, Np, nz), (args.L_perp, args.L_perp, L / args.cells), dtype=torch.complex128, device="cuda")
    Lc = grid.length[2]
    x, y, z = grid.x64
    psi = dipgpe.normalize(grid, (torch.exp(-(x * x + y * y) / 2) * (1 + 0.5 * torch.cos(2 * math.pi * z / Lc))
                              ).to(torch.complex128), n * Lc)
    fracs = [float(f) for f in args.q_fracs.split(",")]
    rows, warm = [], {}
    with dipgpe.runs.Run("tube_crystal_bdg", vars(args)) as run:
        for a_s in [float(v) for v in args.a_list.split(",")]:
            model = cell_model(grid, a_s)
            t0 = time.time()
            psi, info = dipgpe.ground_state(model, psi, tol=1e-10, preconditioner="combined")
            line = (psi.abs() ** 2).sum((0, 1))
            contrast = float((line.max() - line.min()) / (line.max() + line.min()))
            row = {"a_s": a_s, "contrast": contrast, "mu": float(info["mu"]), "gs_converged": bool(info["converged"]),
                   "bands": {}}
            print(f"a_s = {a_s}: crystal contrast {contrast:.4f}, mu {float(info['mu']):.6f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if contrast < 0.05:
                print("  crystal lost: stop", flush=True)
                row["lost"] = True
                rows.append(row)
                break
            amp = psi.abs().to(torch.complex128)                       # real ground state
            kz = grid.k64[2]
            dpsi = torch.fft.ifftn(1j * kz * torch.fft.fftn(amp, dim=grid.dims), dim=grid.dims)
            for f in fracs:
                t0 = time.time()
                w, _, binfo = bogoliubov(model, psi, q=f * math.pi / Lc, n_modes=args.n_modes, tol=args.tol,
                                         max_iter=args.max_iter, x0=warm.get(f) if args.warm else None,
                                         deflate=[dpsi] if f == 0 else None)
                warm[f] = binfo["g"]                                   # warm start for the next a_s
                row["bands"][str(f)] = w.tolist()
                row.setdefault("iterations", {})[str(f)] = binfo["iterations"]
                print(f"  q = {f:g} pi/L: omega = {[round(v, 6) for v in w.tolist()]} "
                      f"({binfo['iterations']} iterations, {time.time() - t0:.0f} s)", flush=True)
            rows.append(row)
            run.save_json("crystal_bdg.json", {"rows": rows})
        # soft mode: lowest q = 0 frequency above the gauge mode (translation deflated)
        pts = [(r["a_s"], r["bands"]["0.0"][1]) for r in rows if "0.0" in r.get("bands", {})]
        out = {"rows": rows, "soft_q0": pts}
        if len(pts) >= 2:
            A = torch.tensor([[1.0, a] for a, _ in pts], dtype=torch.float64)
            y = torch.tensor([[w ** 4] for _, w in pts], dtype=torch.float64)
            c = torch.linalg.lstsq(A, y).solution.reshape(-1)
            out["a_end_from_omega4"] = float(-c[0] / c[1])
            print(f"q = 0 soft mode: omega^4 linear in a_s -> a_end = {out['a_end_from_omega4']:.4f} a0", flush=True)
        run.save_json("crystal_bdg.json", out)
        run.result(**{k: v for k, v in out.items() if k == "a_end_from_omega4"})


if __name__ == "__main__":
    main()

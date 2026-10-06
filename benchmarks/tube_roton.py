"""Roton instability of the uniform 164Dy tube from Bogoliubov spectra (paper C1).

The z-uniform ground state is computed in a short cell, and the lowest
Bogoliubov frequency is scanned over quasi-momenta q; its minimum over q is the
roton frequency omega_rot. omega_rot^2 is linear in a_s near the instability, and
its zero gives the spinodal a_rot* of the uniform state on the given grid.

  uv run python benchmarks/tube_roton.py --density 6250 --a-list 89.9,90.0,90.1 [--dx 0.25] [--L-perp 24]

(Runs before 2026-10-06 were made with the identical `tube_bdg.py roton`; the
run name tube_bdg_roton is kept.)
"""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
import tube_cell  # noqa: E402
from dipgpe.bdg import bogoliubov  # noqa: E402
from tube_cell import RHO, Cell  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402


def roton(args, run):
    if args.L_perp is not None:                       # Cell reads the module constant
        tube_cell.L_PERP = args.L_perp
    rows = []
    for a_s in [float(v) for v in args.a_list.split(",")]:
        cell = Cell(a_s, args.dx, 8)
        # A short cell (reciprocal vector 2 pi / 1.0 >> q) so that the lowest band at q is k = q.
        rho = RHO if args.density is None else UNITS.per_micron(args.density)
        _, grid, psi = cell.solve(rho, 1.0, seed=0.0)
        model = tube_model(grid, a_s)
        qs_ = [args.q_min + i * args.q_step for i in range(int(round((args.q_max - args.q_min) / args.q_step)) + 1)]
        ws = []
        for q in qs_:
            w, _, _ = bogoliubov(model, psi, q=q, n_modes=1)
            ws.append(float(w[0]))
        i = min(range(len(ws)), key=ws.__getitem__)
        lo, hi = max(i - 1, 0), min(i + 1, len(ws) - 1)
        # parabolic refinement of the minimum in q
        if 0 < i < len(ws) - 1:
            y0, y1, y2 = ws[i - 1], ws[i], ws[i + 1]
            d = (y0 - 2 * y1 + y2)
            dq = 0.5 * args.q_step * (y0 - y2) / d if d > 0 else 0.0
            w_min = y1 - 0.25 * (y0 - y2) * dq / args.q_step
            q_min = qs_[i] + dq
        else:
            w_min, q_min = ws[i], qs_[i]
        rows.append({"a_s": a_s, "q": qs_, "omega": ws, "omega_rot": w_min, "k_rot": q_min,
                     "roton_wavelength_l": 2 * math.pi / q_min})
        print(f"a_s = {a_s}: omega_rot = {w_min:.6f} at k = {q_min:.4f} (wavelength {2 * math.pi / q_min:.3f} l)",
              flush=True)
    A = torch.tensor([[1.0, r["a_s"]] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["omega_rot"] ** 2 for r in rows], dtype=torch.float64).reshape(-1, 1)
    c = torch.linalg.lstsq(A, y).solution.reshape(-1)
    a_rot = float(-c[0] / c[1])
    print(f"omega_rot^2 linear in a_s -> zero at a_rot* = {a_rot:.4f} a0 ", flush=True)
    run.save_json("roton.json", {"rows": rows, "a_rot": a_rot})
    run.result(a_rot=a_rot)


def add_arguments(r):
    r.add_argument("--a-list", default="92.4,92.5,92.6")
    r.add_argument("--dx", type=float, default=0.25)
    r.add_argument("--density", type=float, default=None, help="atoms per um (default 2500)")
    r.add_argument("--L-perp", type=float, default=None, help="transverse box (default 24 l)")
    r.add_argument("--q-min", type=float, default=1.30)
    r.add_argument("--q-max", type=float, default=1.60)
    r.add_argument("--q-step", type=float, default=0.025)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args()
    args.cmd = "roton"
    with dipgpe.runs.Run("tube_bdg_roton", vars(args)) as run:
        roton(args, run)


if __name__ == "__main__":
    main()

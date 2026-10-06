"""Ground states shown in Fig. 1(d) of paper C1: the uniform superfluid and the
supersolid crystal at 6250/um and the crystal of isolated droplets at 700/um,
on the grid and with the model of the 3D ramps (one cell, dx = 0.25 l,
L_perp = 24 l, complex128). Saves the column densities n(y, z) = int dx |psi|^2
(dipoles along y) and the line densities as a recorded run.

  uv run python benchmarks/tube_states.py
"""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_crystal_bdg import cell_model  # noqa: E402
from tube_ramp import UNITS  # noqa: E402

STATES = [("uniform", 6250.0, 2.76, 92.0, False),
          ("supersolid", 6250.0, 2.76, 89.5, True),
          ("droplets", 700.0, 3.2, 83.0, True)]


def main():
    um = UNITS.microns(1.0)
    dx, L_perp = 0.25, 24.0
    Np = 2 * round(L_perp / dx / 2)
    out = {}
    with dipgpe.runs.Run("tube_states", {"states": STATES, "dx": dx, "L_perp": L_perp}) as run:
        for name, density, cell_um, a_s, modulated in STATES:
            Lc = cell_um / um
            nz = round(Lc / dx)
            grid = dipgpe.Grid((Np, Np, nz), (L_perp, L_perp, Lc), dtype=torch.complex128, device="cuda")
            x, y, z = grid.x64
            seed = torch.exp(-(x * x + y * y) / 2) * (1 + (0.5 * torch.cos(2 * math.pi * z / Lc) if modulated else 0))
            n = UNITS.per_micron(density)
            psi, info = dipgpe.ground_state(cell_model(grid, a_s), dipgpe.normalize(grid, seed.to(torch.complex128), n * Lc),
                                        tol=1e-9, preconditioner="combined")
            rho = psi.abs() ** 2
            col = (rho.sum(0) * grid.dx[0]).cpu()                       # n(y, z), per l^2
            line = (rho.sum((0, 1)) * grid.dx[0] * grid.dx[1]).cpu()   # n(z), per l
            contrast = float((line.max() - line.min()) / (line.max() + line.min()))
            fs = float(Lc ** 2 / (line.sum() * grid.dx[2] * (grid.dx[2] / line).sum()))
            out[name] = {"column": col, "line": line, "y": grid.x64[1].reshape(-1).cpu(),
                         "z": grid.x64[2].reshape(-1).cpu(), "a_s": a_s, "density": density, "cell_um": cell_um,
                         "contrast": contrast, "fs": fs, "l_um": 1.0 / um}
            print(f"{name}: {density:g}/um, a_s = {a_s} a0, cell {cell_um} um, contrast {contrast:.3f}, "
                  f"Leggett f_s {fs:.3f} ({info['iterations']} iterations)", flush=True)
        torch.save(out, run.data("states.pt"))
        run.result(**{k: {"contrast": v["contrast"], "fs": v["fs"]} for k, v in out.items()})


if __name__ == "__main__":
    main()

"""Unit cell of the 164Dy tube (3D, transverse box L_PERP, dipoles along y): ground
states of one periodic cell with a fixed number of z points, continued from the
previous state of the same cell. Shared by the tube benchmarks."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402

N_PER_UM, L_PERP = 2500.0, 24.0
RHO = UNITS.per_micron(N_PER_UM)                      # atoms per l at the default density


class Cell:
    """Ground states of one unit cell with a fixed number of z points (the transverse
    box is the module constant L_PERP, read when the cell is created)."""

    def __init__(self, a_s, dx, N_z):
        self.a_s, self.dx, self.N_z = a_s, dx, N_z
        self.L_perp = L_PERP
        self.Np = int(round(self.L_perp / dx))
        self.psi = None

    def grid(self, a):
        return dipgpe.Grid((self.Np, self.Np, self.N_z), (self.L_perp, self.L_perp, a), dtype=torch.complex128,
                           device="cuda")

    def solve(self, rho, a, q=0.0, seed=0.4):
        """Energy per unit length; continues from the previous state of this cell."""
        grid = self.grid(a)
        terms = tube_model(grid, self.a_s).terms + ((dipgpe.Twist(q),) if q else ())
        model = dipgpe.GPE(grid, *terms)
        if self.psi is None:
            x, y, z = grid.x64
            self.psi = (torch.exp(-(x * x + y * y) / 2) * (1 + seed * torch.cos(2 * math.pi * z / a))).to(grid.dtype)
        psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, self.psi, rho * a))
        if not q:
            self.psi = psi
        return float(dipgpe.energy(model, psi)) / a, grid, psi

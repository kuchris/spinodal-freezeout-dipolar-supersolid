"""Sample test run (sample input and output): exercises the main features
on small problems and prints each result next to its reference.

    uv run python examples/sample_run.py [--device cpu]

Expected output (RTX 5070 Ti, complex128 unless noted) is in
examples/sample_run_expected.txt; numbers should agree to the digits that the
"reference" column states, and the run takes about a minute on a GPU (longer on
a CPU). Every run is recorded under runs/ (see README, "Run records").

1. 2D harmonic ground state (minimizer): E = 1 (d/2 per atom).
2. Kohn mode (split step, Strang): <x>(t) = x0 cos t.
3. Bogoliubov spectrum of a uniform 1D gas at Bloch q: w = sqrt(e (e + 2 g n)),
   e = (q + 2 pi j / L)^2 / 2.
4. Cylindrical vs spherical dipolar cut-off for a flat cloud: equal energies.
5. 164Dy tube supersolid (contact + dipolar + LHY, tube cut-off), one 2.69 um
   unit cell at n = 2500 /um, a_s = 91 a0 (below a_rot* = 92.32 a0): modulated
   ground state. A small, coarse cell (L_perp = 12 l, dx = 0.25 l) for speed;
   its reference is the expected output, not a converged physical value.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dipgpe  # noqa: E402
from dipgpe.units import OscillatorUnits  # noqa: E402

C128 = torch.complex128


def harmonic_ground_state(device):
    grid = dipgpe.Grid((128, 128), 16.0, dtype=C128, device=device)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)))
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.3, 2.0)))
    return float(dipgpe.energy(model, psi)), info["iterations"]


def kohn_mode(device, x0=0.5, T=2 * math.pi):
    grid = dipgpe.Grid((128, 128), 16.0, dtype=C128, device=device)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(50.0))
    psi, _ = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5)))
    psi = dipgpe.translate(grid, psi, (x0, 0.0))
    steps = 2000
    stepper = dipgpe.SplitStep(model, T / steps)
    err = 0.0
    for k in range(1, 9):
        psi = stepper.evolve(psi, steps // 8)
        t = k * T / 8
        err = max(err, abs(float(dipgpe.center_of_mass(grid, psi)[0]) - x0 * math.cos(t)) / x0)
    return err


def uniform_bogoliubov(device, g=1.0, n=1.0, L=10.0, q=0.3):
    grid = dipgpe.Grid((64,), L, dtype=C128, device=device)
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    psi0 = torch.full(grid.shape, math.sqrt(n), dtype=C128, device=device)
    omega, _, _ = dipgpe.bdg.bogoliubov(model, psi0, q=q, n_modes=4, tol=1e-8)
    e = sorted(0.5 * (q + 2 * math.pi * j / L) ** 2 for j in range(-4, 5))
    exact = sorted(math.sqrt(x * (x + 2 * g * n)) for x in e)[:4]
    return omega.tolist(), exact


def cylinder_vs_sphere(device):
    def energy(grid, cutoff):
        x, y, z = grid.x64
        rho = (100.0 * torch.exp(-(x * x + y * y) / (2 * 1.2 ** 2) - z * z / (2 * 0.8 ** 2))).to(grid.real_dtype)
        d = dipgpe.Dipolar(grid, 0.3, direction=(0, 0, 1), cutoff=cutoff)
        return float(grid.integrate(0.5 * d.potential(rho) * rho))
    e_sphere = energy(dipgpe.Grid((64, 64, 64), 16.0, dtype=C128, device=device), "sphere")
    e_cyl = energy(dipgpe.Grid((64, 64, 40), (16.0, 16.0, 10.0), dtype=C128, device=device), "cylinder")
    return e_cyl, e_sphere


def tube_supersolid(device, a_s_bohr=91.0, n_per_um=2500.0, cell_um=2.69, dx=0.25, L_perp=12.0):
    units = OscillatorUnits(164, 2 * math.pi * 150)
    a_s, a_dd = units.bohr(a_s_bohr), units.bohr(130.8)
    L = cell_um / units.microns(1.0)
    n_perp, n_z = int(round(L_perp / dx)), int(round(L / dx))
    grid = dipgpe.Grid((n_perp, n_perp, n_z), (L_perp, L_perp, L), dtype=C128, device=device)
    x, y, z = grid.x64
    model = dipgpe.GPE(grid, dipgpe.Potential((0.5 * (x * x + y * y)).expand(grid.shape).to(grid.real_dtype)),
                   dipgpe.Contact(4 * math.pi * a_s),
                   dipgpe.Dipolar(grid, 4 * math.pi * a_dd, direction=(0, 1, 0), cutoff="tube"),
                   dipgpe.LHY(dipgpe.lhy_coefficient(a_s, a_dd)))
    N = units.per_micron(n_per_um) * L
    psi0 = (torch.exp(-(x * x + y * y) / 2) * (1 + 0.4 * torch.cos(2 * math.pi * z / L))).to(C128)
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, psi0, N), tol=1e-8)
    line = (psi.abs() ** 2).sum((0, 1))
    contrast = float((line.max() - line.min()) / (line.max() + line.min()))
    return float(dipgpe.energy(model, psi)) / N, contrast, info["iterations"], list(grid.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    with dipgpe.runs.Run("sample_run", vars(args)) as run:
        t0 = time.time()
        E, it = harmonic_ground_state(args.device)
        print(f"1. harmonic 2D ground state: E/N = {E:.12f} (reference 1, error {abs(E - 1):.1e}; {it} iterations)")
        err = kohn_mode(args.device)
        print(f"2. Kohn mode over one period: max |<x> - x0 cos t| / x0 = {err:.1e} (reference: < 1e-4)")
        w, exact = uniform_bogoliubov(args.device)
        print("3. uniform 1D Bogoliubov at q = 0.3: " + ", ".join(f"{a:.8f}" for a in w)
              + "\n   analytic:                          " + ", ".join(f"{a:.8f}" for a in exact)
              + f"\n   max relative error {max(abs(a - b) / b for a, b in zip(w, exact)):.1e}")
        e_cyl, e_sph = cylinder_vs_sphere(args.device)
        print(f"4. dipolar energy, cylinder cut-off {e_cyl:.10f} vs sphere {e_sph:.10f} "
              f"(relative difference {abs(e_cyl / e_sph - 1):.1e}; reference < 1e-6)")
        e, c, it, shape = tube_supersolid(args.device)
        print(f"5. 164Dy tube, n = 2500/um, a_s = 91.0 a0, grid {shape}: E/N = {e:.8f} hbar w, "
              f"line contrast {c:.4f} ({it} iterations; reference: sample_run_expected.txt, modulated below "
              f"a_rot* = 92.32 a0)")
        print(f"total {time.time() - t0:.1f} s")
        run.result(harmonic_E=E, kohn_error=err, bogoliubov=w, bogoliubov_exact=exact,
                   dipolar_cylinder=e_cyl, dipolar_sphere=e_sph, tube_E_per_N=e, tube_contrast=c)


if __name__ == "__main__":
    main()

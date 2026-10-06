"""Superfluid-to-crystal transition of a 164Dy gas in an infinite tube (Phase 4 gate).

Model (extended GPE): transverse trap omega_x = omega_y = 2 pi x 150 Hz, free along
the tube axis z (periodic unit cell of length L), dipoles along y, a_dd = 130.8 a0,
average linear density n, LHY with the exact Q5(eps_dd). The dipolar kernel is cut
at transverse distance R = L_perp / 2 (Dipolar(cutoff="tube")).

Literature, quoted verbatim (docs/ISSUES.md P4-3). Blakie et al., arXiv:2004.12577,
reduced theory (transverse Gaussian ansatz): "a* ≈ 91.6 a0" (continuous) at
n = 2.5×10³/µm; "a_rot* = 81.2 a0" at n = 0.7×10³/µm; "a_rot* = 88.6 a0" at
n = 6.25×10³/µm. Smith, Baillie & Blakie, arXiv:2212.07607: "the transition
boundary a* predicted by the reduced theory is approximately 1 a0 to 2 a0 lower
than that of the full eGPE calculation", and in the tube the intermediate-density
transition is continuous with a* = a_rot*.

Method (default): roton softening. For the z-uniform ground state, a small
modulation eps cos(k z) is evolved in imaginary time; its amplitude decays as
exp(-lambda tau), lambda being the lowest eigenvalue of the energy Hessian in that
k sector. lambda changes sign at the roton instability; a_rot* is the highest
zero crossing over the cell lengths L = 2 pi / k. This avoids imaginary-time
critical slowing near the transition (crystal ground states took 3-11 min each,
the softening rate 15 s). ``--crystal`` instead measures the contrast of the
crystalline ground state (slow) as a cross-check.

Usage:
  python benchmarks/tube_supersolid.py [--output results/tube_supersolid.json]
  python benchmarks/tube_supersolid.py --crystal
  python benchmarks/tube_supersolid.py --live 8765      # watch at http://localhost:8765
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
import dipgpe  # noqa: E402
from dipgpe.units import OscillatorUnits  # noqa: E402

UNITS = OscillatorUnits(164, 2 * math.pi * 150)
A_DD = UNITS.bohr(130.8)


def tube_model(grid, a_s_bohr):
    a_s = UNITS.bohr(a_s_bohr)
    x, y, _ = grid.x64
    return dipgpe.GPE(grid, dipgpe.Potential((0.5 * (x * x + y * y)).expand(grid.shape).to(grid.real_dtype)),
                  dipgpe.Contact(4 * math.pi * a_s),
                  dipgpe.Dipolar(grid, 4 * math.pi * A_DD, direction=(0, 1, 0), cutoff="tube"),
                  dipgpe.LHY(dipgpe.lhy_coefficient(a_s, A_DD)))


def softening_rate(a_s_bohr, n_per_um, L, dx=0.125, L_perp=24.0, N_z=8, dt=0.002, tau=8.0,
                   eps=1e-5, chunks=40, dtype=torch.complex128, live=None, state=None):
    """Decay rate lambda of a cos(2 pi z / L) modulation of the uniform tube state
    (negative: the uniform state is unstable to crystallization at this L).
    A dict ``state`` receives the grid and the uniform ground state."""
    N_perp = int(round(L_perp / dx))
    grid = dipgpe.Grid((N_perp, N_perp, N_z), (L_perp, L_perp, L), dtype=dtype, device="cuda")
    model = tube_model(grid, a_s_bohr)
    x, y, z = grid.x64
    N = UNITS.per_micron(n_per_um) * L
    psi0 = dipgpe.normalize(grid, torch.exp(-(x * x + y * y) / 2 + 0 * z).to(dtype), N)
    psi0, _ = dipgpe.ground_state(model, psi0, tol=1e-10, live=live)
    if state is not None:
        state.update(grid=grid, psi=psi0)
    k = 2 * math.pi / L
    psi = dipgpe.normalize(grid, psi0 * (1 + eps * torch.cos(k * z)).to(dtype), N)
    stepper = dipgpe.SplitStep(model, dt, imaginary=True)
    phase = torch.exp(-1j * k * z).to(dtype)
    times, logs = [], []
    steps = round(tau / dt / chunks)
    for i in range(chunks):
        psi = stepper.evolve(psi, steps)
        rho = psi.abs() ** 2
        times.append((i + 1) * steps * dt)
        logs.append(math.log(float((rho * phase).sum().abs() / rho.sum())))
        if live is not None:
            live.report({"imaginary_time": times[-1], "log_modulation": logs[-1]}, grid, psi)
    t, a = torch.tensor(times), torch.tensor(logs)
    late = t >= tau / 2
    A = torch.stack([t[late], torch.ones(int(late.sum()), dtype=t.dtype)], 1)
    return -torch.linalg.lstsq(A, a[late, None]).solution[0, 0].item()


def roton_softening(n_per_um, a_lo, a_hi, cells, live=None, done=0, **kw):
    """Highest zero crossing of lambda(a_s) over the cells, by linear interpolation
    between a_lo and a_hi. Returns (a_rot, L, rows)."""
    rows, best = [], None
    for L in cells:
        rates = []
        for a_s in (a_lo, a_hi):
            state = {}
            if live is not None:
                live.stage(f"n = {n_per_um:.0f}/um, a_s = {a_s} a0, cell L = {L} l: uniform state, then modulation decay")
            rates.append(softening_rate(a_s, n_per_um, L, live=live, state=state, **kw))
            done += 1
            if live is not None:   # frame: the z-uniform state whose stability lambda measures
                live.checkpoint(done, {"lambda": rates[-1], "a_s": a_s, "L": L}, state["grid"], state["psi"])
        l_lo, l_hi = rates
        a = a_lo + (a_hi - a_lo) * (-l_lo) / (l_hi - l_lo)
        rows.append({"L": L, "L_um": UNITS.microns(L), "lambda_lo": l_lo, "lambda_hi": l_hi, "a_cross": a})
        print(f"  n = {n_per_um:.0f}/um, L = {L:.2f} l ({UNITS.microns(L):.2f} um): lambda({a_lo}) = {l_lo:+.5f}, "
              f"lambda({a_hi}) = {l_hi:+.5f} -> {a:.3f} a0", flush=True)
        if best is None or a > best[0]:
            best = (a, L)
    return best[0], best[1], rows


def tube_ground_state(a_s_bohr, n_per_um, L, L_perp=24.0, dx=0.125, dtype=torch.complex128,
                      seed=0.4, tol=1e-9, live=None, state=None):
    """Ground state in one unit cell; returns (energy per atom, axis contrast, line contrast, iterations).
    A dict ``state`` receives the grid and the ground state."""
    a_s = UNITS.bohr(a_s_bohr)
    N_perp, N_z = int(round(L_perp / dx)), max(8, int(round(L / dx)))
    grid = dipgpe.Grid((N_perp, N_perp, N_z), (L_perp, L_perp, L), dtype=dtype, device="cuda")
    x, y, z = grid.x64
    model = dipgpe.GPE(grid, dipgpe.Potential((0.5 * (x * x + y * y)).expand(grid.shape).to(grid.real_dtype)),
                   dipgpe.Contact(4 * math.pi * a_s),
                   dipgpe.Dipolar(grid, 4 * math.pi * A_DD, direction=(0, 1, 0), cutoff="tube"),
                   dipgpe.LHY(dipgpe.lhy_coefficient(a_s, A_DD)))
    N = UNITS.per_micron(n_per_um) * L
    psi0 = (torch.exp(-(x * x + y * y) / 2) * (1 + seed * torch.cos(2 * math.pi * z / L))).to(dtype)
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, psi0, N), tol=tol, live=live)
    if state is not None:
        state.update(grid=grid, psi=psi)
    rho = psi.abs() ** 2
    axis = rho[N_perp // 2, N_perp // 2, :]
    line = rho.sum((0, 1))
    contrast = lambda f: float((f.max() - f.min()) / (f.max() + f.min()))
    return float(dipgpe.energy(model, psi)) / N, contrast(axis), contrast(line), info["iterations"]


def tiled(grid, psi, copies):
    """The periodic one-cell state repeated ``copies`` times along z (display only)."""
    big = dipgpe.Grid((grid.shape[0], grid.shape[1], grid.shape[2] * copies),
                  (grid.length[0], grid.length[1], grid.length[2] * copies), dtype=grid.dtype, device=psi.device)
    return big, psi.repeat(1, 1, copies)


DENSITIES = [  # n (1/um), reduced-theory literature value, bracket and cells for the scan
    {"n": 700.0, "reduced": 81.2, "kind": "a_rot*", "a_lo": 84.0, "a_hi": 84.75, "cells": [4.5, 5.0, 5.5]},
    {"n": 2500.0, "reduced": 91.6, "kind": "a* = a_rot*", "a_lo": 92.25, "a_hi": 92.5,
     "cells": [4.2, 4.35, 4.5, 4.65, 4.8]},
    {"n": 6250.0, "reduced": 88.6, "kind": "a_rot*", "a_lo": 89.5, "a_hi": 90.25, "cells": [3.5, 4.0, 4.5]},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crystal", action="store_true", help="contrast of crystal ground states at n = 2500/um")
    ap.add_argument("--output", type=Path, default=None, help="extra copy of the result JSON (the run directory always keeps one)")
    ap.add_argument("--live", type=int, nargs="?", const=8765, default=None, metavar="PORT",
                    help="serve live progress and previews at http://localhost:PORT")
    args = ap.parse_args()
    with dipgpe.runs.Run("tube_supersolid_crystal" if args.crystal else "tube_supersolid", vars(args)) as run:
        return _run(args, run)


def _run(args, run):
    print(f"l = {UNITS.length * 1e6:.4f} um, a_dd = {A_DD:.5f} l")
    crystal_a = (91.5, 92.0, 92.25, 92.5)
    tasks = len(crystal_a) if args.crystal else sum(2 * len(d["cells"]) for d in DENSITIES)
    if args.live is not None and not args.crystal:
        tasks += 1   # crystal illustration after the scan
    live = None
    if args.live is not None:
        live = run.live("164Dy tube supersolid", tasks, args.live, progress_label="task")
        print(f"live view: {live.url}")
    if args.crystal:
        scan = []
        for i, a in enumerate(crystal_a):
            t0 = time.time()
            if live is not None:
                live.stage(f"crystal ground state, n = 2500/um, a_s = {a} a0, cell 4.5 l")
            state = {}
            E, C_axis, C_line, iters = tube_ground_state(a, 2500.0, 4.5, live=live, state=state)
            if live is not None:
                live.checkpoint(i + 1, {"contrast": C_axis, "energy": E, "a_s": a}, *tiled(state["grid"], state["psi"], 3))
            scan.append({"a_s": a, "energy": E, "C_axis": C_axis, "iterations": iters})
            print(f"a_s = {a:.2f} a0, L = 4.5: E/N = {E:.8f}, C = {C_axis:.4f} ({iters} iterations, "
                  f"{time.time() - t0:.0f} s)", flush=True)
        result = {"method": "crystal", "L": 4.5, "scan": scan}
    else:
        rows, done = [], 0
        for d in DENSITIES:
            a_rot, L, cells = roton_softening(d["n"], d["a_lo"], d["a_hi"], d["cells"], live=live, done=done)
            done += 2 * len(d["cells"])
            rows.append({**d, "a_rot_full": a_rot, "L_rot": L, "L_rot_um": UNITS.microns(L),
                         "offset": a_rot - d["reduced"], "cell_scan": cells})
            print(f"n = {d['n']:.0f}/um: full eGPE a_rot* = {a_rot:.2f} a0 (roton wavelength "
                  f"{UNITS.microns(L):.2f} um); reduced theory {d['kind']} = {d['reduced']} a0; "
                  f"offset {a_rot - d['reduced']:+.2f} a0 (literature: full-eGPE a* about 1-2 a0 above the "
                  f"reduced a*; a_rot* = a* only for the continuous case)", flush=True)
        result = {"method": "roton softening", "densities": rows}
        if live is not None:
            # Illustration for the viewer only, not part of the gate: one crystal
            # ground state below a_rot*, shown as three periods along the tube.
            a, L, state = 92.0, 4.5, {}
            live.stage(f"illustration (not part of the gate): crystal ground state, n = 2500/um, "
                       f"a_s = {a} a0, cell {L} l, shown as 3 periods")
            t0 = time.time()
            E, C_axis, _, iters = tube_ground_state(a, 2500.0, L, live=live, state=state)
            live.checkpoint(done + 1, {"contrast": C_axis, "energy": E, "a_s": a, "L": L},
                            *tiled(state["grid"], state["psi"], 3))
            print(f"illustration: crystal at a_s = {a} a0, C = {C_axis:.3f} ({iters} iterations, "
                  f"{time.time() - t0:.0f} s)", flush=True)
            result["illustration"] = {"a_s": a, "n": 2500.0, "L": L, "C_axis": C_axis, "energy": E,
                                      "note": "viewer illustration, not part of the gate"}
    output = run.save_json("tube_supersolid.json", result, also=args.output)
    if "densities" in result:
        run.result(a_rot={str(int(d["n"])): d["a_rot_full"] for d in result["densities"]})
    if live is not None:
        live.close()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

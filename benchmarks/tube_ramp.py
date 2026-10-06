"""Scattering-length ramps across the superfluid-supersolid transition of the
164Dy tube (paper C1), in the quasi-1D model or the full 3D eGPE.

Protocol (Kirkby, Salman, Gasenzer, Chomaz, PRR 7 (2025)): uniform ground state at
a_i, truncated-Wigner noise, hold for --equil-ms at a_i, then a_s(t) = a_i +
(a_f - a_i)(t - t_eq)/tau_Q, then hold at a_f for --hold-ms. Recorded every
--record-ms for every realization: the Leggett bound f_s = L^2 / (N int dz / n)
of the line density n(z) and its contrast; the sample-averaged g2(dz); line
densities of the first --save-samples realizations. Freeze-out: the time after
the critical crossing t_c (from --a-c) at which <f_s> first drops below 0.98.

Their anchor (n = 2500/um): quasi-1D, closed kernel, l = 1.08 um, eta = 4.25,
L = 344 um (128 cells), a: 96 -> 88 a0, a_c = 91.05 a0, tau_Q = 1 - 770 ms;
freeze-out ~ tau_Q^0.352(3) (thermal) / 0.346(2) (quantum noise only).

  uv run python benchmarks/tube_ramp.py --model 1d --tau-ms 50,100 --samples 32
  uv run python benchmarks/tube_ramp.py --model 3d --cells 16 --tau-ms 20 --samples 1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dipgpe  # noqa: E402
from dipgpe.noise import wigner_noise  # noqa: E402
from dipgpe.quasi1d import tube_model  # noqa: E402
from dipgpe.units import OscillatorUnits  # noqa: E402

UNITS = OscillatorUnits(164, 2 * math.pi * 150)
A_DD = UNITS.bohr(130.8)
MS = UNITS.omega * 1e-3            # 1 ms in units of 1/omega
KB, HBAR = 1.380649e-23, 1.054571817e-34


def a_of_t(args):
    t_eq, tau = args.equil_ms * MS, args.tau * MS
    def a_bohr(t):
        if t <= t_eq:
            return args.a_i
        return args.a_i + (args.a_f - args.a_i) * min(1.0, (t - t_eq) / tau)
    return a_bohr


def build(args):
    """(grid, model, psi0 (single uniform state), line_density(psi), tube_axis)."""
    um = UNITS.microns(1.0)
    L = args.cells * args.cell_um / um
    n = UNITS.per_micron(args.density)
    dtype = torch.complex64 if args.complex64 else torch.complex128
    a_t = a_of_t(args)
    if args.model == "1d":
        N_z = 2 * round(L / args.dx / 2)
        grid = dipgpe.Grid(N_z, L, dtype=dtype, device="cuda")
        model = tube_model(grid, lambda t: UNITS.bohr(a_t(t)), A_DD, args.ell_um / um, args.eta, kernel=args.kernel)
        psi0 = torch.full(grid.shape, math.sqrt(n), dtype=dtype, device="cuda")
        line = lambda psi: psi.abs() ** 2
        return grid, model, psi0, line, 0
    # 3D: tube along z, dipoles along y, isotropic transverse trap (omega = 1)
    Np = 2 * round(args.L_perp / args.dx / 2)
    N_z = 2 * round(L / args.dx / 2) if args.initial == "uniform" else args.cells * round(L / args.cells / args.dx)
    def model_on(g, a_fn):
        x, y, _ = g.x64
        a = (lambda t: UNITS.bohr(a_fn(t)))
        return dipgpe.GPE(g, dipgpe.Potential((0.5 * (x * x + y * y)).expand(g.shape).to(g.real_dtype)),
                      dipgpe.Contact(lambda t: 4 * math.pi * a(t)),
                      dipgpe.Dipolar(g, 4 * math.pi * A_DD, direction=(0, 1, 0), cutoff="tube"),
                      dipgpe.LHY(lambda t: dipgpe.lhy_coefficient(a(t), A_DD)))
    grid = dipgpe.Grid((Np, Np, N_z), (args.L_perp, args.L_perp, L), dtype=dtype, device="cuda")
    if args.initial == "uniform":
        cell = dipgpe.Grid((Np, Np, 8), (args.L_perp, args.L_perp, 8 * args.dx), dtype=torch.complex128, device="cuda")
        x, y, _ = cell.x64
        seed = torch.exp(-(x * x + y * y) / 2).expand(cell.shape).to(torch.complex128)
        psi_c, _ = dipgpe.ground_state(model_on(cell, lambda t: args.a_i), dipgpe.normalize(cell, seed, n * cell.length[2]),
                                   tol=1e-9, preconditioner="combined")
        psi0 = psi_c.mean(dim=2, keepdim=True).expand(grid.shape).to(dtype).contiguous()      # z-uniform
    else:                                    # crystal: one cell of N_z / cells points, tiled
        nz = N_z // args.cells
        if nz * args.cells != N_z:
            raise ValueError("tube points must be a multiple of the cell count (adjust --dx or --cell-um)")
        cell = dipgpe.Grid((Np, Np, nz), (args.L_perp, args.L_perp, L / args.cells), dtype=torch.complex128,
                       device="cuda")
        x, y, z = cell.x64
        seed = (torch.exp(-(x * x + y * y) / 2) * (1 + 0.5 * torch.cos(2 * math.pi * z / cell.length[2]))
                ).to(torch.complex128)
        psi_c, info = dipgpe.ground_state(model_on(cell, lambda t: args.a_i), dipgpe.normalize(cell, seed, n * cell.length[2]),
                                      tol=1e-9, preconditioner="combined")
        line_c = (psi_c.abs() ** 2).sum((0, 1))
        print(f"  crystal cell at {args.a_i} a0: line contrast "
              f"{float((line_c.max() - line_c.min()) / (line_c.max() + line_c.min())):.3f}", flush=True)
        psi0 = psi_c.repeat(1, 1, args.cells).to(dtype).contiguous()
    line = lambda psi: (psi.abs() ** 2).sum(dim=(-3, -2)) * grid.dx[0] * grid.dx[1]
    return grid, model_on(grid, a_t), psi0, line, 2


def count_peaks(line_n, threshold=0.2):
    """Local maxima of the (periodic) line density above (1 + threshold) x its mean."""
    left, right = torch.roll(line_n, 1, -1), torch.roll(line_n, -1, -1)
    high = line_n > (1 + threshold) * line_n.mean(-1, keepdim=True)
    return ((line_n > left) & (line_n >= right) & high).sum(-1)


def observables(line_n, dz, L):
    """f_s (Leggett), contrast per sample, and g2(dz) summed over samples."""
    N = line_n.sum(-1) * dz
    fs = L * L / (N * (dz / line_n.clamp_min(1e-30)).sum(-1))
    contrast = (line_n.amax(-1) - line_n.amin(-1)) / (line_n.amax(-1) + line_n.amin(-1))
    nk = torch.fft.rfft(line_n, dim=-1)
    corr = torch.fft.irfft(nk * nk.conj(), n=line_n.shape[-1], dim=-1) / line_n.shape[-1]
    g2 = corr / (line_n.mean(-1, keepdim=True) ** 2)
    return fs, contrast, g2.sum(0)


def run_tau(args, tau_ms, run):
    args.tau = tau_ms
    grid, model, psi0, line, axis = build(args)
    L = grid.length[axis]
    n_mu = None
    if args.noise == "thermal":
        kT = KB * args.T_nK * 1e-9 / (HBAR * UNITS.omega)
        noise, info = wigner_noise(grid, kT, axis=axis, cutoff=2 * kT * args.noise_cutoff, samples=args.samples,
                                   seed=args.seed + int(tau_ms * 1000))
    else:                    # quantum: |k| < 1/xi, xi = 1/sqrt(mu) (3D: mu minus the transverse zero point)
        n_mu = float(dipgpe.chemical_potential(model, psi0, 0.0)) - (1.0 if args.model == "3d" else 0.0)
        noise, info = wigner_noise(grid, 0.0, axis=axis, k_max=math.sqrt(abs(n_mu)) * args.noise_cutoff, samples=args.samples,
                                   seed=args.seed + int(tau_ms * 1000))
    N = UNITS.per_micron(args.density) * L
    psi = psi0.unsqueeze(0) + noise
    dt_max = dipgpe.max_stable_dt(grid) * args.dt_safety
    rec = args.record_ms * MS
    sub = max(1, math.ceil(rec / dt_max))
    dt = rec / sub
    stepper = dipgpe.SplitStep(model, dt, keep_norm=args.complex64)
    t_end = (args.equil_ms + tau_ms + args.hold_ms) * MS
    n_rec = round(t_end / rec)
    dz = grid.dx[axis]
    times, fs_all, c_all, g2_all, saved, g2_times, peaks = [], [], [], [], [], [], []
    deep, fs_avg = [], []                       # thermal-robust order parameters
    window = max(1, round(1.0 / args.record_ms))                    # 1 ms running average of n(z)
    recent = []
    t0 = time.time()
    for r in range(n_rec + 1):
        if r:
            psi = stepper.evolve(psi, sub)
        ln = line(psi).to(torch.float64)
        fs, c, g2 = observables(ln, dz, L)
        times.append(stepper.t / MS)
        fs_all.append(fs.cpu())
        c_all.append(c.cpu())
        peaks.append(count_peaks(ln).cpu())
        deep.append(count_peaks(ln, threshold=0.5).cpu())
        recent.append(ln)
        if len(recent) > window:
            recent.pop(0)
        fs_avg.append(observables(torch.stack(recent).mean(0), dz, L)[0].cpu())
        if r % args.store_every == 0:
            g2_times.append(stepper.t / MS)
            g2_all.append((g2 / args.samples)[: ln.shape[-1] // 2].to(torch.float32).cpu())
            saved.append(ln[: args.save_samples].to(torch.float32).cpu())
        if r == 1:
            per = (time.time() - t0) / sub
            print(f"    {per * 1e3:.2f} ms per step ({args.samples} samples), {sub * n_rec} steps, "
                  f"estimated {per * sub * n_rec:.0f} s", flush=True)
    fs_all, c_all = torch.stack(fs_all), torch.stack(c_all)               # (records, samples)
    mean_fs = fs_all.mean(1)
    t_c = args.equil_ms + tau_ms * (args.a_i - args.a_c) / (args.a_i - args.a_f) if args.a_c else None
    cross = None
    if t_c is not None:
        melt = args.a_f > args.a_i                                   # reverse ramp: crystal -> uniform
        after = [i for i, t in enumerate(times) if t >= t_c and (float(mean_fs[i]) > 0.98 if melt
                                                                 else float(mean_fs[i]) < 0.98)]
        cross = times[after[0]] - t_c if after else None
    seconds = time.time() - t0
    stem = f"tau_{tau_ms:g}ms"
    torch.save({"times_ms": times, "fs": fs_all, "contrast": c_all, "peaks": torch.stack(peaks),
                "deep_peaks": torch.stack(deep), "fs_avg_1ms": torch.stack(fs_avg), "g2": torch.stack(g2_all), "g2_times_ms": g2_times,
                "line": torch.stack(saved), "dz_um": UNITS.microns(dz), "args": vars(args)}, run.data(stem + ".pt"))
    res = {"tau_ms": tau_ms, "freeze_out_ms": cross, "t_c_ms": t_c, "samples": args.samples,
           "noise_modes": info["modes"], "noise_added_fraction": info["mean_added_atoms"] / N,
           "dt": dt, "steps": sub * n_rec, "seconds": seconds, "final_mean_fs": float(mean_fs[-1]),
           "final_mean_contrast": float(c_all[-1].mean()), "final_mean_peaks": float(peaks[-1].float().mean()),
           "ground_state_cells": args.cells}
    run.result(**{stem: res})
    print(f"  tau_Q = {tau_ms:g} ms: freeze-out {cross if cross is None else round(cross, 3)} ms after t_c, "
          f"final <f_s> = {res['final_mean_fs']:.3f}, noise adds {100 * res['noise_added_fraction']:.2f}% "
          f"({seconds:.0f} s)", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=["1d", "3d"], default="1d")
    p.add_argument("--density", type=float, default=2500.0, help="atoms per um")
    p.add_argument("--a-i", type=float, default=96.0)
    p.add_argument("--a-f", type=float, default=88.0)
    p.add_argument("--a-c", type=float, default=91.05, help="critical a_s for the freeze-out time (0: none)")
    p.add_argument("--tau-ms", default="100")
    p.add_argument("--cells", type=int, default=128)
    p.add_argument("--cell-um", type=float, default=2.6875, help="344 um / 128")
    p.add_argument("--dx", type=float, default=None, help="grid spacing in l (default 0.0625 1D, 0.25 3D)")
    p.add_argument("--L-perp", type=float, default=24.0)
    p.add_argument("--ell-um", type=float, default=1.08)
    p.add_argument("--eta", type=float, default=4.25)
    p.add_argument("--kernel", choices=["exact", "closed"], default="closed")
    p.add_argument("--noise", choices=["quantum", "thermal"], default="quantum")
    p.add_argument("--T-nK", type=float, default=20.0)
    p.add_argument("--noise-cutoff", type=float, default=1.0,
                   help="scale the noise cutoff: thermal energy cutoff 2 kT, quantum k_max 1/xi (sensitivity tests)")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equil-ms", type=float, default=10.0)
    p.add_argument("--hold-ms", type=float, default=None, help="default: tau_Q / 2")
    p.add_argument("--record-ms", type=float, default=0.5)
    p.add_argument("--dt-safety", type=float, default=0.9)
    p.add_argument("--save-samples", type=int, default=4)
    p.add_argument("--store-every", type=int, default=1, help="store g2 and line densities every n records")
    p.add_argument("--complex64", action="store_true")
    p.add_argument("--initial", choices=["uniform", "crystal"], default="uniform",
                   help="crystal: ground-state cell at a_i tiled --cells times (reverse ramps, a_f > a_i)")
    args = p.parse_args()
    if args.dx is None:
        args.dx = 0.0625 if args.model == "1d" else 0.25
    with dipgpe.runs.Run(f"tube_ramp_{args.model}", vars(args)) as run:
        out = []
        for tau in [float(v) for v in args.tau_ms.split(",")]:
            hold = args.hold_ms
            args.hold_ms = tau / 2 if hold is None else hold
            out.append(run_tau(args, tau, run))
            args.hold_ms = hold
        run.save_json("summary.json", out)


if __name__ == "__main__":
    main()

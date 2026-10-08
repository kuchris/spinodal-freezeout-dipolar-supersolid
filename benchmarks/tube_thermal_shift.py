"""Thermal effects on the onset and on the roton in the truncated-Wigner simulations (paper C1).

Sanchez-Baena et al., Nat. Commun. 14, 1868 (2023) and Phys. Rev. Research 6,
023183 (2024), showed that thermal fluctuations soften the roton of a dipolar
condensate and move the superfluid-supersolid transition to larger a_s. The
classical-field (truncated-Wigner) simulations contain the thermally populated
low-energy modes explicitly, so part of this effect could be present in the
thermal runs of paper C1, whose reference points (a_rot*, a*) are
zero-temperature values. Two tests:

  ramps  Median onset (f_s < threshold) after the zero-temperature spinodal for
         thermal and quantum-noise forward ramps at several tau_Q, for the
         thresholds 0.98, 0.9, 0.7 and 0.5. A rigid shift of the onset curve (for
         instance a shifted instability) would give the same advance for every
         threshold. Ramp times whose median onset falls after the end of the
         linear ramp are excluded from the exponents (paper C1, Appendix D).
         Medians use the upper middle realization for an even count and errors
         are bootstrap standard deviations over 2000 resamples, as in
         tube_ramp_errors.py (paper C1, Appendix D).
  holds  Line densities n(z, t) of holds above the metastable window (no crystal
         branch, no nucleation), with thermal noise and, as the reference, with
         quantum noise only (linear fluctuations: zero-temperature frequencies,
         same analysis). For each tube mode k_m = 2 pi m / L the realization-
         averaged autocorrelation Re<c_m(t + tau) c_m*(t)> of the Fourier
         amplitude is fitted by A exp(-gamma tau) cos(omega tau); bootstrap over
         realizations gives the uncertainty. Modes within one step of the
         zero-temperature roton minimum that are weakly damped at both
         temperatures are compared mode by mode; with omega_rot^2 linear in a_s
         and zero at a_rot* (Bogoliubov, paper C1), s = omega_rot,0^2 /
         (a_hold - a_rot*) converts the mean change of omega^2 into an effective
         shift, delta a_rot = -<omega(T)^2 - omega(0)^2> / s. Three
         autocorrelation windows test the sensitivity to the fit.

Usage:
  uv run python benchmarks/tube_thermal_shift.py --holds 0:<run>,5:<run>,10:<run> --a-hold 90.8 \\
      --ramps-thermal 20:<run>,50:<run>+<run>,100:<run>+<run>,200:<run> \\
      --ramps-quantum 20:<run>,50:<run>,100:<run>,200:<run>
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dipgpe  # noqa: E402

OMEGA = 2 * math.pi * 150.0                      # omega_perp (1/s)
A_ROT = 89.845                                   # zero-temperature spinodal at 6250/um (paper C1)


def load(run_id, tau):
    """Time series of one tau_Q from runs/<id>/data (full) or data_C1/<id> (reduced;
    always with C1_ARCHIVE_ONLY set, to test the public-release path)."""
    full = ROOT / "runs" / f"{run_id}_tube_ramp_3d" / "data" / f"tau_{tau:g}ms.pt"
    if full.exists() and not os.environ.get("C1_ARCHIVE_ONLY"):
        d = torch.load(full, map_location="cpu", weights_only=False)
        return {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in d.items() if k != "args"}
    d = np.load(ROOT / "data_C1" / run_id / f"tau_{tau:g}ms.npz")
    return {k: d[k] for k in d.files}


def added_fraction(run_id):
    rj = ROOT / "runs" / f"{run_id}_tube_ramp_3d" / "result.json"
    return json.loads(rj.read_text())["tau_20ms"]["noise_added_fraction"] if rj.exists() else None


class DampedCosineFit:
    """Least-squares fit of A exp(-g tau) cos(w tau) on a (g, w) grid, A linear."""

    def __init__(self, tau, w=np.arange(0.15, 0.80, 0.0005), g=np.arange(0.0, 0.16, 0.002)):
        self.w, self.g = w, g
        self.B = np.exp(-g[:, None, None] * tau) * np.cos(w[None, :, None] * tau)
        self.BB = (self.B ** 2).sum(-1)

    def __call__(self, C):
        y = C / C[0]
        A = (self.B * y).sum(-1) / self.BB
        res = ((A[..., None] * self.B - y) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(res), res.shape)
        return float(self.w[j]), float(self.g[i])


def acf(x, nlag):
    n = x.shape[0]
    return np.array([np.mean((x[lag:] * np.conj(x[:n - lag])).real) for lag in range(nlag)])


def hold_modes(run_id, ms, t_from, nlag=100, boots=40):
    d = load(run_id, 20)
    t, line = d["g2_times_ms"].astype(float), d["line"].astype(float)
    keep = t >= t_from
    tt = (t[keep] - t[keep][0]) * 1e-3 * OMEGA
    c = np.fft.fft(line[keep] - line[keep].mean(-1, keepdims=True), axis=-1)
    fit = DampedCosineFit(np.arange(nlag) * (tt[1] - tt[0]))
    rng = np.random.default_rng(0)
    rows = []
    for m in ms:
        x = c[:, :, m]
        w, g = fit(acf(x, nlag))
        bs = [fit(acf(x[:, rng.integers(0, x.shape[1], x.shape[1])], nlag))[0] for _ in range(boots)]
        rows.append({"m": int(m), "omega": w, "omega_err": float(np.std(bs)), "damping": g,
                     "power": float(np.mean(np.abs(x) ** 2))})
    L_um = line.shape[-1] * float(d["dz_um"])
    return {"run": run_id, "frames": int(keep.sum()), "samples": int(line.shape[1]), "L_um": L_um, "modes": rows,
            "added_fraction": added_fraction(run_id), "nlag": nlag}


def holds(args):
    ms = list(range(args.m_min, args.m_max + 1))
    entries = [e.split(":") for e in args.holds.split(",")]
    out = {"a_hold": args.a_hold, "a_rot": A_ROT, "reference": args.reference, "windows": {}}
    for nlag in (60, 100, 150):
        res = {T: hold_modes(r, ms, args.t_from, nlag=nlag) for T, r in entries}
        ref = res[args.reference]
        w0 = np.array([r["omega"] for r in ref["modes"]])
        j = int(np.argmin(w0))
        s = w0[j] ** 2 / (args.a_hold - A_ROT)            # d omega_rot^2 / d a_s with zero at a_rot*
        win = {"roton_m": ms[j], "omega_rot_T0": float(w0[j]), "slope": float(s), "temperatures": res, "shifts": {}}
        for T, r in res.items():
            if T == args.reference:
                continue
            near = [i for i in range(len(ms)) if abs(i - j) <= 1 and r["modes"][i]["damping"] < 0.05
                    and ref["modes"][i]["damping"] < 0.05]
            if near:
                d2 = np.array([r["modes"][i]["omega"] ** 2 - ref["modes"][i]["omega"] ** 2 for i in near])
                e2 = np.array([2 * r["modes"][i]["omega"] * max(r["modes"][i]["omega_err"], 2e-3) for i in near])
                wt = 1 / e2 ** 2
                da = float(-(wt * d2).sum() / wt.sum() / s)
                da_err = float(wt.sum() ** -0.5 / s)
            else:
                da = da_err = None
            rel = [(x["omega"] - y["omega"]) / y["omega"] for x, y in zip(r["modes"], ref["modes"])]
            win["shifts"][T] = {"modes_used": [ms[i] for i in near], "delta_a_rot": da, "delta_a_rot_err": da_err,
                                "relative_shift_per_mode": rel,
                                "damping_near_roton": [r["modes"][i]["damping"] for i in range(len(ms)) if abs(i - j) <= 2]}
            print(f"window {nlag}: T = {T} nK: modes {[ms[i] for i in near]} -> delta a_rot = "
                  f"{da if da is None else round(da, 4)} +- {da_err if da_err is None else round(da_err, 4)} a0",
                  flush=True)
        out["windows"][str(nlag)] = win
    return out


def median(v):
    """Upper middle value for an even count, as in tube_ramp_errors.py (paper C1,
    Appendix D); realizations that never cross count as the latest."""
    v = np.sort(np.where(np.isnan(v), np.inf, v))
    return float(v[len(v) // 2])


def onsets(run_ids, tau, thr):
    out = []
    for r in run_ids:
        d = load(r, tau)
        t, fs = d["times_ms"].astype(float), d["fs"].astype(float)
        ts = 10 + tau * (92 - A_ROT) / 4
        for s in range(fs.shape[1]):
            i = np.where(fs[:, s] < thr)[0]
            out.append(t[i[0]] - ts if len(i) else np.nan)
    return np.array(out)


def ramps(args):
    parse = lambda s: {float(k): v.split("+") for k, v in (e.split(":") for e in s.split(","))}  # noqa: E731
    th, qu = parse(args.ramps_thermal), parse(args.ramps_quantum)
    rng = np.random.default_rng(0)
    out = {}
    for thr in (0.98, 0.9, 0.7, 0.5):
        rows, samples = [], []
        for tau in sorted(set(th) & set(qu)):
            a, b = onsets(qu[tau], tau, thr), onsets(th[tau], tau, thr)
            end = tau * (1 - (92 - A_ROT) / 4)                   # end of the linear ramp, after the spinodal
            post = bool(max(median(a), median(b)) > end)
            d = median(a) - median(b)
            bs = [median(rng.choice(a, a.size)) - median(rng.choice(b, b.size)) for _ in range(2000)]
            rows.append({"tau_ms": tau, "quantum_median": median(a), "thermal_median": median(b),
                         "n_quantum": int(a.size), "n_thermal": int(b.size), "advance_ms": d,
                         "advance_err": float(np.std(bs)), "post_ramp": post})
            samples.append((a, b, post))
        use = [i for i, r in enumerate(rows) if not r["post_ramp"]]
        logt = np.log([rows[i]["tau_ms"] for i in use])
        slope = lambda med: float(np.polyfit(logt, np.log(med), 1)[0])  # noqa: E731
        ex = {}
        for key, k in (("quantum", 0), ("thermal", 1)):
            med = [median(samples[i][k]) for i in use]
            bs = [slope([median(rng.choice(samples[i][k], samples[i][k].size)) for i in use])
                  for _ in range(2000)]
            ex[key] = {"exponent": slope(med), "err": float(np.std(bs))}
        out[str(thr)] = {"rows": rows, "exponents_from_T0_spinodal": ex,
                         "tau_used": [rows[i]["tau_ms"] for i in use]}
        print(f"f_s < {thr}: advances " + ", ".join(
            f"{r['tau_ms']:g} ms: {r['advance_ms']:.2f} +- {r['advance_err']:.2f}{' (post-ramp)' if r['post_ramp'] else ''}"
            for r in rows) + f"; exponents quantum {ex['quantum']['exponent']:.3f} +- {ex['quantum']['err']:.3f}, "
            f"thermal {ex['thermal']['exponent']:.3f} +- {ex['thermal']['err']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holds", default=None, help="T:run_id pairs of holds (tau_Q 20 ms into a_hold); T = 0: quantum noise")
    ap.add_argument("--reference", default="0", help="label of the zero-temperature hold")
    ap.add_argument("--a-hold", type=float, default=90.8)
    ap.add_argument("--t-from", type=float, default=40.0, help="start of the analysed hold window (ms)")
    ap.add_argument("--m-min", type=int, default=22)
    ap.add_argument("--m-max", type=int, default=40)
    ap.add_argument("--ramps-thermal", default=None, help="tau:run[+run] pairs")
    ap.add_argument("--ramps-quantum", default=None, help="tau:run[+run] pairs")
    args = ap.parse_args()
    with dipgpe.runs.Run("tube_thermal_shift", vars(args)) as run:
        result = {}
        if args.holds:
            result["holds"] = holds(args)
        if args.ramps_thermal and args.ramps_quantum:
            result["ramps"] = ramps(args)
        run.save_json("tube_thermal_shift.json", result)
        summary = {}
        if "holds" in result:
            summary["delta_a_rot"] = {n: {T: v["delta_a_rot"] for T, v in w["shifts"].items()}
                                      for n, w in result["holds"]["windows"].items()}
        if "ramps" in result:
            summary["advance_ms"] = {thr: {r["tau_ms"]: r["advance_ms"] for r in v["rows"]}
                                     for thr, v in result["ramps"].items()}
            summary["exponents"] = {thr: v["exponents_from_T0_spinodal"] for thr, v in result["ramps"].items()}
        run.result(**summary)


if __name__ == "__main__":
    main()

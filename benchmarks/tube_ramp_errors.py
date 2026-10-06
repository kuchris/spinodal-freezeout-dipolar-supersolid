"""Error budget of the freeze-out exponents (paper C1): threshold robustness,
bootstrap over realizations and systematic shifts of the reference point.

For every data set (density, direction) and threshold f_th, the per-realization
crossing times after the reference point are computed, the median per tau_Q is
fitted with a power law in log-log, and
  - bootstrap: realizations are resampled with replacement within each tau_Q
    (B times), giving the spread of the fitted zeta;
  - reference: the reference scattering length is shifted by +-delta_a;
  - threshold: zeta for f_th = 0.99, 0.98, 0.95, 0.90 (melting: 0.90-0.99 upward).
Ramp window: the linear-ramp fit uses only the tau_Q whose median onset
(f_th = 0.98) comes before the end of the ramp; onsets during the final hold
(fast ramps that overshoot, 6250/um at 10 ms, 700/um at 10 and 20 ms) are
kept as observations but not fitted. The threshold test uses one common set of
tau_Q for which the medians of all four thresholds precede the ramp end, and
the difference between the primary and common-window slopes enters the error
as a fit-window term. Total = bootstrap, half the threshold range, reference
shift, fit-window shift and a 0.015 spatial allowance (measured for 6250
forward, used as a proxy elsewhere) in quadrature; these are sensitivity
estimates, not confidence intervals.
Writes errors.json and fig_zeta_threshold.png as a recorded run.

  uv run python benchmarks/tube_ramp_errors.py [--boot 2000] [--delta-a 0.01]
"""

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dipgpe import runs  # noqa: E402
import tube_ramp_figures as F  # noqa: E402

THRESHOLDS = [0.99, 0.98, 0.95, 0.90]


def per_realization(D, tau, ref, thr):
    a = D["args"]
    t = D["times_ms"]
    t_ref = a["equil_ms"] + tau * (a["a_i"] - ref) / (a["a_i"] - a["a_f"])
    up = a["a_f"] > a["a_i"]
    fs = D["fs"]
    out = []
    for s in range(fs.shape[1]):
        col = fs[:, s].tolist()
        k = next((i for i in range(len(t)) if t[i] >= t_ref and (col[i] > thr if up else col[i] < thr)), None)
        out.append(t[k] - t_ref if k is not None else None)
    return out


def ramp_window(data, ref):
    """(primary taus, common taus): median onset before the ramp end at f_th = 0.98,
    and for every threshold."""
    primary, common = [], []
    for tau in sorted(data):
        a = data[tau][0]["args"]
        t_ref = a["equil_ms"] + tau * (a["a_i"] - ref) / (a["a_i"] - a["a_f"])
        left = a["equil_ms"] + tau - t_ref                         # ramp time left after the reference
        meds = {}
        for thr in THRESHOLDS:
            vals = [x for D in data[tau] for x in per_realization(D, tau, ref, thr) if x is not None and x > 0]
            meds[thr] = median(vals) if vals else math.inf
        if meds[0.98] <= left + 1e-8:
            primary.append(tau)
        if all(m <= left + 1e-8 for m in meds.values()):
            common.append(tau)
    return primary, common


def fit(taus, ys):
    lx = [math.log(x) for x in taus]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)


def median(v):
    v = sorted(v)
    return v[len(v) // 2]


def dataset(ids):
    data = {}
    for rid in ids:
        for tau, D in F.load(rid).items():
            data.setdefault(tau, []).append(D)
    return data


def zeta_of(data, ref, thr):
    taus, meds, per = [], [], {}
    for tau in sorted(data):
        vals = [x for D in data[tau] for x in per_realization(D, tau, ref, thr) if x is not None and x > 0]
        if len(vals) >= 2:
            taus.append(tau)
            meds.append(median(vals))
            per[tau] = vals
    return (fit(taus, meds) if len(taus) >= 2 else float("nan")), per


def bootstrap(per, B, rng):
    taus = sorted(per)
    zs = []
    for _ in range(B):
        meds = [median([rng.choice(per[t]) for _ in per[t]]) for t in taus]
        zs.append(fit(taus, meds))
    zs.sort()
    m = sum(zs) / B
    sd = math.sqrt(sum((z - m) ** 2 for z in zs) / (B - 1))
    return sd, zs[int(0.16 * B)], zs[int(0.84 * B)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--delta-a", type=float, default=0.01, help="reference shift (a0) for the systematic error")
    args = p.parse_args()
    sets = [("6250 forward", F.FORWARD[6250], F.SPINODAL[6250]),
            ("2500 forward", F.FORWARD[2500], F.SPINODAL[2500]),
            ("700 forward", F.FORWARD[700], F.SPINODAL[700]),
            ("6250 reverse", F.REVERSE[6250], F.BRANCH_END[6250])]
    rng = random.Random(1)
    out = {}
    with runs.Run("tube_ramp_errors", vars(args)) as run:
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        for name, ids, ref in sets:
            data = dataset(ids)
            primary, common = ramp_window(data, ref)
            sub = lambda keep: {t: data[t] for t in keep}  # noqa: E731
            z0, per = zeta_of(sub(primary), ref, 0.98)
            stat = bootstrap(per, args.boot, rng)[0]
            rows = {}
            for thr in THRESHOLDS:
                z, per_t = zeta_of(sub(common), ref, thr)
                sd, lo, hi = bootstrap(per_t, args.boot, rng)
                rows[thr] = {"zeta": z, "boot_sd": sd, "boot_16": lo, "boot_84": hi,
                             "realizations": {str(t): len(v) for t, v in per_t.items()}}
            z_plus, _ = zeta_of(sub(primary), ref + args.delta_a, 0.98)
            z_minus, _ = zeta_of(sub(primary), ref - args.delta_a, 0.98)
            z_all, _ = zeta_of(data, ref, 0.98)
            thr_spread = max(r["zeta"] for r in rows.values()) - min(r["zeta"] for r in rows.values())
            ref_spread = abs(z_plus - z_minus) / 2
            window = abs(z0 - rows[0.98]["zeta"])
            total = math.sqrt(stat ** 2 + (thr_spread / 2) ** 2 + ref_spread ** 2 + window ** 2 + 0.015 ** 2)
            out[name] = {"thresholds": {str(k): v for k, v in rows.items()}, "zeta_0.98": z0,
                         "primary_taus": primary, "common_taus": common, "zeta_all_points": z_all,
                         "stat_bootstrap": stat, "syst_threshold_half_range": thr_spread / 2,
                         "syst_reference": ref_spread, "syst_fit_window": window, "syst_grid": 0.015,
                         "total": total, "reference": ref, "delta_a": args.delta_a,
                         "zeta_ref_plus_0.03": zeta_of(sub(primary), ref + 0.03, 0.98)[0]}
            print(f"{name}: zeta(0.98) = {z0:.4f} over tau {primary} (all points {z_all:.3f}); bootstrap "
                  f"{stat:.4f}; thresholds over {common}: "
                  + ", ".join(f"{t}: {rows[t]['zeta']:.3f}" for t in THRESHOLDS)
                  + f"; reference +-{args.delta_a} a0 -> +-{ref_spread:.4f}; window {window:.4f}; "
                  f"total {total:.4f}", flush=True)
            ax.errorbar(THRESHOLDS, [rows[t]["zeta"] for t in THRESHOLDS],
                        yerr=[rows[t]["boot_sd"] for t in THRESHOLDS], marker="o", ms=3, capsize=2, label=name)
        ax.axhline(1 / 3, color="k", ls=":", lw=1, label="1/3")
        ax.set_xlabel("freeze-out threshold f_th")
        ax.set_ylabel("ζ")
        ax.invert_xaxis()
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(run.file("fig_zeta_threshold.png"), dpi=200)
        run.save_json("errors.json", out)


if __name__ == "__main__":
    main()

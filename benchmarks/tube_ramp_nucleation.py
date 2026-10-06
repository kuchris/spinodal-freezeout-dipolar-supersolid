"""Nucleation rate per unit length and time from hold simulations (paper C1), and
the predicted nucleation probability during ramps.

Hold runs (tube_ramp.py with --a-c 0, thermal noise): for each realization the
first time t_1 at which a deep droplet (> 1.5 x mean line density; > 1.2 x for
older runs without the deep count) appears, measured from the moment the ramp
crosses a* (nucleation is impossible above a*). Realizations without a droplet
are right-censored at the end of the run. For a Poisson process along a tube of
length L the waiting time is exponential with rate Gamma L, and the maximum-
likelihood estimate is
    Gamma L = (number of nucleated realizations) / (total exposure time),
with a relative statistical error 1/sqrt(number of events). The survival
curve S(t) is written out to check the exponential form.

Prediction for a ramp crossing the window between a* and the spinodal at rate
r (a0/ms): expected number of nuclei m = L int Gamma(a_s) da / r, with Gamma
interpolated linearly in a_s between the measured holds and Gamma(a*) = 0; the
probability that the ramp nucleates before the spinodal is 1 - exp(-m).
(t_1 includes a growth time to a detectable droplet, so Gamma here is an
effective rate of detectable nuclei.)

  uv run python benchmarks/tube_ramp_nucleation.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dipgpe import runs  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_data import load_tau_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs"
A_STAR, A_ROT = 90.256, 89.845


def save(fig, run, stem):
    """PNG for viewing and vector PDF for the manuscript."""
    fig.savefig(run.file(stem + ".png"), dpi=200)
    fig.savefig(run.file(stem + ".pdf"))


def holds():
    out = []
    for d in sorted(ROOT.glob("*_tube_ramp_3d")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        p = meta.get("params", {})
        if p.get("a_c") == 0 and p.get("density") == 6250 and p.get("noise") == "thermal" \
                and meta.get("status") == "finished":
            try:
                D = next(iter(load_tau_files(d).values()))
            except (FileNotFoundError, StopIteration):
                continue
            out.append((d.name, p, D))
    return out


def roton_power(D, t_ms):
    """Sample-averaged line-density structure factor summed over the roton band
    (Fourier indices 28-38 of the 88 um tube, wavelengths 2.3-3.2 um), from the
    stored g2(dz), at the record closest to t_ms."""
    g = D["g2"].double()
    full = torch.cat([g, g[:, -1:], g[:, 1:].flip(1)], 1)                          # even extension
    S = torch.fft.rfft(full, dim=1).real[:, 28:39].sum(1)
    tg = D["g2_times_ms"]
    return float(S[min(range(len(tg)), key=lambda i: abs(tg[i] - t_ms))])


def ramps():
    """5 nK thermal ramps through the window at 6250/um (a_c != 0, a_f below the spinodal)."""
    out = []
    for d in sorted(ROOT.glob("*_tube_ramp_3d")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        p = meta.get("params", {})
        if p.get("a_c") and p.get("density") == 6250 and p.get("noise") == "thermal" and p.get("T_nK") == 5.0                 and p.get("a_i", 0) > A_STAR and p.get("a_f", 99) < A_ROT                 and p.get("noise_cutoff", 1.0) == 1.0 and meta.get("status") == "finished":
            try:
                files = load_tau_files(d)
            except FileNotFoundError:
                continue
            for tau, D in sorted(files.items()):
                out.append((d.name, p, float(D["args"]["tau"]), D))
    return out


def ramp_test(integral, L_um):
    """Fraction of realizations with a droplet before the spinodal crossing, against
    1 - exp(-L int Gamma da / r). This is a lower bound on the nucleated fraction:
    a nucleus formed late in the window is detected only after the crossing, where
    it cannot be told apart from the spinodal route."""
    rows = []
    for name, p, tau, D in ramps():
        t = D["times_ms"]
        a_i, a_f = p["a_i"], p["a_f"]
        t_at = lambda a: p["equil_ms"] + tau * (a_i - a) / (a_i - a_f)  # noqa: E731
        t_star, t_rot = t_at(A_STAR), t_at(A_ROT)
        deep = D.get("deep_peaks")
        counts = deep if deep is not None else D["peaks"]
        rel = []
        for s in range(counts.shape[1]):
            col = counts[:, s].tolist()
            k = next((i for i in range(len(t)) if t[i] >= t_star and col[i] >= 1), None)
            rel.append(t[k] - t_rot if k is not None else None)
        n = len(rel)
        early = sum(1 for x in rel if x is not None and x < 0)
        rate = (a_i - a_f) / tau                                                       # a0 per ms
        tau_eq = 4.0 / rate                                                            # tau_Q of a 92 -> 88 ramp
        m = L_um * integral / rate * 1e-3
        s_star = roton_power(D, t_star)
        rows.append({"run": name, "a_i": a_i, "a_f": a_f, "tau_ms": tau, "S_rot_at_a_star": s_star, "tau_Q_equivalent_ms": tau_eq, "deep_count": deep is not None,
                     "realizations": n, "early": early,
                     "P_predicted": 1 - math.exp(-m), "first_droplet_rel_spinodal_ms": rel})
        print(f"ramp {name} {a_i:g} -> {a_f:g} a0 in {tau:g} ms (tau_Q-equivalent {tau_eq:.0f} ms, "
              f"{'deep' if deep is not None else 'loose'}): droplet before the spinodal in {early}/{n}; "
              f"predicted P = {1 - math.exp(-m):.2f}; roton power at a* {s_star:.3f}", flush=True)
    pooled = {}                                         # same protocol, several runs
    for r in rows:
        key = (r["a_i"], round(r["tau_Q_equivalent_ms"]), r["deep_count"])
        e = pooled.setdefault(key, {"early": 0, "realizations": 0, "P_predicted": r["P_predicted"]})
        e["early"] += r["early"]
        e["realizations"] += r["realizations"]
    for (a_i, tau_eq, deep), e in sorted(pooled.items()):
        print(f"pooled ramps from {a_i:g} a0, tau_Q-equivalent {tau_eq} ms ({'deep' if deep else 'loose'}): "
              f"{e['early']}/{e['realizations']} before the spinodal, predicted {e['P_predicted']:.2f}", flush=True)
    return rows


def first_times(p, D):
    t = D["times_ms"]
    tau = p["tau_ms"] if isinstance(p["tau_ms"], (int, float)) else float(str(p["tau_ms"]).split(",")[0])
    t_star = p["equil_ms"] + tau * (p["a_i"] - A_STAR) / (p["a_i"] - p["a_f"])
    deep = D.get("deep_peaks")
    counts = deep if deep is not None else D["peaks"]
    out = []
    for s in range(counts.shape[1]):
        col = counts[:, s].tolist()
        k = next((i for i in range(len(t)) if t[i] >= t_star and col[i] >= 1), None)
        out.append((t[k] - t_star) if k is not None else None)
    return out, t[-1] - t_star, deep is not None


def estimate(t1, t_end, L_um):
    """MLE of Gamma (per um per s) for a pure exponential and for an exponential with
    a lag t0 (= earliest event, the MLE of the shift), both with right censoring."""
    events = [x for x in t1 if x is not None]
    n_cens = len(t1) - len(events)
    exposure = sum(events) + t_end * n_cens
    gamma = len(events) / exposure / L_um * 1e3 if exposure > 0 else float("nan")
    t0 = min(events) if events else 0.0
    exp_lag = sum(x - t0 for x in events) + (t_end - t0) * n_cens
    gamma_lag = len(events) / exp_lag / L_um * 1e3 if exp_lag > 0 else float("nan")
    err = 1 / math.sqrt(len(events)) if events else float("nan")
    return {"realizations": len(t1), "nucleated": len(events), "Gamma_per_um_s": gamma,
            "Gamma_err": gamma * err, "lag_ms": t0, "Gamma_lag_per_um_s": gamma_lag,
            "Gamma_lag_err": gamma_lag * err,
            "median_t1_ms": sorted(events)[len(events) // 2] if events else None}


def main():
    with runs.Run("tube_ramp_nucleation", {}) as run:
        rows, pooled = [], {}
        for name, p, D in holds():
            t1, t_end, deep = first_times(p, D)
            L_um = p["cells"] * p["cell_um"]
            r = estimate(t1, t_end, L_um)
            r["S_rot_at_hold_start"] = roton_power(D, p["equil_ms"] + float(str(p["tau_ms"]).split(",")[0]))
            rows.append({"run": name, "a_s": p["a_f"], "T_nK": p["T_nK"], "deep_count": deep,
                         "noise_cutoff": p.get("noise_cutoff", 1.0), "a_i": p["a_i"], **r})
            print(f"{name}: a_s {p['a_f']:.2f}, T {p['T_nK']:g} nK, {r['nucleated']}/{r['realizations']} "
                  f"nucleated, Gamma = {r['Gamma_per_um_s']:.3g} +- {r['Gamma_err']:.2g} per um per s "
                  f"({'deep' if deep else 'loose'} count, from {p['a_i']:g} a0, roton power at the hold start "
                  f"{r['S_rot_at_hold_start']:.3f})", flush=True)
            key = (p["a_f"], p["T_nK"], deep, p.get("noise_cutoff", 1.0), p["a_i"])
            P = pooled.setdefault(key, {"t1": [], "t_end": t_end, "L_um": L_um, "runs": []})
            P["t1"] += t1
            P["t_end"] = min(P["t_end"], t_end)
            P["runs"].append(name)
        table = []
        for (a_s, T, deep, cut, a_i), P in sorted(pooled.items()):
            r = estimate(P["t1"], P["t_end"], P["L_um"])
            table.append({"a_s": a_s, "T_nK": T, "deep_count": deep, "noise_cutoff": cut, "a_i": a_i,
                          "runs": P["runs"], **r})
            print(f"pooled a_s {a_s:.2f}, T {T:g} nK ({'deep' if deep else 'loose'}, cutoff x{cut:g}, "
                  f"from {a_i:g} a0): "
                  f"{r['nucleated']}/{r['realizations']}, Gamma = {r['Gamma_per_um_s']:.3g} +- "
                  f"{r['Gamma_err']:.2g}; with lag {r['lag_ms']:.1f} ms: {r['Gamma_lag_per_um_s']:.3g} +- "
                  f"{r['Gamma_lag_err']:.2g} per um per s; median t1 {r['median_t1_ms']}", flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
        panels = [(axes[0], lambda e: e["T_nK"] == 5.0, lambda e: f"{e['a_s']:.2f} a$_0$"),
                  (axes[1], lambda e: e["a_s"] == 90.05, lambda e: f"{e['T_nK']:g} nK")]
        for ax, sel, lab in panels:
            for e in table:
                if not (e["deep_count"] and e["noise_cutoff"] == 1.0 and e["a_i"] == 92.0 and sel(e)):
                    continue
                P = pooled[(e["a_s"], e["T_nK"], True, 1.0, 92.0)]
                ts = sorted(x for x in P["t1"] if x is not None)
                n = len(P["t1"])
                line, = ax.step([0] + ts + [P["t_end"]], [1] + [1 - (i + 1) / n for i in range(len(ts))]
                                + [1 - len(ts) / n], where="post", label=f"{lab(e)}, n={n}")
                rate = e["Gamma_per_um_s"] * P["L_um"] * 1e-3
                tt = torch.linspace(0, P["t_end"], 100)
                ax.plot(tt, torch.exp(-rate * tt), ls=":", color=line.get_color(), lw=1)
            ax.set_yscale("log")
            ax.set_ylim(0.03, 1.1)
            ax.set_xlabel("t − t(a*) (ms)")
            ax.legend(fontsize=6)
        axes[0].set_ylabel("fraction without a deep droplet S(t)")
        axes[0].set_title("(c) T = 5 nK", fontsize=8)
        axes[1].set_title("(c) a$_s$ = 90.05 a$_0$", fontsize=8)
        fig.tight_layout()
        save(fig, run, "fig_survival")
        # prediction at 5 nK for the thermal ramps (92 -> 88 a0, rate 4/tau_Q a0/ms)
        pts = sorted({(e["a_s"], e["Gamma_per_um_s"]) for e in table if e["T_nK"] == 5.0 and e["deep_count"]
                      and e["noise_cutoff"] == 1.0 and e["a_i"] == 92.0})
        grid = [(A_STAR, 0.0)] + sorted(pts, reverse=True)
        L_um = 32 * 2.76
        pred = {}
        xs = [g[0] for g in grid] + [A_ROT]
        ys = [g[1] for g in grid] + [grid[-1][1]]                                       # flat below the last hold
        integral = sum(0.5 * (y0 + y1) * abs(x1 - x0)                                   # per um per s * a0
                       for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])))
        for tau in (20, 50, 100, 200, 400, 800):
            r = 4.0 / tau                                                               # a0 per ms
            m = L_um * integral / r * 1e-3                                              # expected nuclei
            pred[tau] = {"expected_nuclei": m, "P_nucleate_before_spinodal": 1 - math.exp(-m)}
            print(f"5 nK ramp tau_Q = {tau} ms: expected nuclei in the window {m:.2f}, "
                  f"P(nucleation before the spinodal) = {1 - math.exp(-m):.2f}", flush=True)
        tau_x = 4.0 * math.log(2) / (L_um * integral * 1e-3)
        print(f"crossover (P = 1/2) at tau_Q = {tau_x:.0f} ms", flush=True)
        tests = ramp_test(integral, L_um)
        run.save_json("nucleation.json", {"holds": rows, "pooled": table, "prediction_5nK": pred,
                                          "tau_cross_ms": tau_x, "grid": grid, "ramp_tests": tests})


if __name__ == "__main__":
    main()

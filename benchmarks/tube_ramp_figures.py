"""Figures for paper C1 from recorded tube_ramp runs (each figure is recorded as
a run: `uv run python benchmarks/tube_ramp_figures.py`).

  fig_freezeout   t_hat vs tau_Q (median over realizations), referred to the
                  spinodal (main) and to a* (inset), all densities, forward and reverse
  fig_dynamics    6250/um: <f_s>(t) for several tau_Q and a line-density kymograph
  fig_hysteresis  6250/um: a_s at which <f_s> crosses 0.98 going down and up vs tau_Q
  fig_nucleation  holds inside the window: droplets vs time for several a_s (5 nK)
                  and temperatures (90.05 a0)
  fig_droplets    700/um: final droplet number and per-realization t_hat spread vs tau_Q
  fig_coexistence 6250/um: double tangent, coexistence window, localized state (static runs)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dipgpe import runs  # noqa: E402
from c1_data import load_states, load_tau_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs"
SPINODAL = {700: 84.506, 2500: 92.314, 6250: 89.845}       # a_rot* on the dynamical grid
A_STAR = {700: 85.28, 2500: 92.314, 6250: 90.256}
BRANCH_END = {6250: 90.39}
FORWARD = {2500: ["20261005-140819", "20261006-012712"], 6250: ["20261005-132617", "20261005-133951"],
           700: ["20261005-145757", "20261005-151305", "20261005-152429", "20261005-171904", "20261006-022814"]}
REVERSE = {6250: ["20261005-155046", "20261006-014539"]}
COLORS = {700: "#c0504d", 2500: "#4f81bd", 6250: "#9bbb59"}


def save(fig, run, stem):
    """PNG for viewing and vector PDF for the manuscript."""
    fig.savefig(run.file(stem + ".png"), dpi=200)
    fig.savefig(run.file(stem + ".pdf"))


def load(run_id):
    """{tau: observables} from the run record or the reduced archive (c1_data)."""
    return load_tau_files(next(ROOT.glob(run_id + "_tube_ramp_3d")))


def pooled(ids):
    """{tau: [D, ...]} over several runs (extra realizations of the same tau_Q)."""
    out = {}
    for rid in ids:
        for tau, D in load(rid).items():
            out.setdefault(tau, []).append(D)
    return out


def pooled_crossings(Ds, tau, ref, threshold=0.98):
    return sorted(x for D in Ds for x in crossings(D, tau, ref, threshold))


def a_of_t(a, tau):
    return lambda t: a["a_i"] + (a["a_f"] - a["a_i"]) * min(1.0, max(0.0, (t - a["equil_ms"]) / tau))


def crossings(D, tau, ref, threshold=0.98):
    """Per-realization time of the f_s threshold crossing after the reference point."""
    a = D["args"]
    t = torch.tensor(D["times_ms"])
    t_ref = a["equil_ms"] + tau * (a["a_i"] - ref) / (a["a_i"] - a["a_f"])
    up = a["a_f"] > a["a_i"]
    out = []
    for s in range(D["fs"].shape[1]):
        f = D["fs"][:, s]
        k = next((i for i in range(len(t)) if t[i] >= t_ref and (f[i] > threshold if up else f[i] < threshold)), None)
        if k is not None:
            out.append(float(t[k]) - t_ref)
    return sorted(out)


def median(x):
    return x[len(x) // 2]


def onsets(ids, ref, threshold=0.98):
    """[(tau, median delay after ref, onset during the ramp?, actual a_s at the onset)]:
    the actual a_s follows the piecewise ramp-and-hold schedule, so an onset in the
    final hold sits at a_f (not on the continued ramp line)."""
    rows = []
    for tau, Ds in sorted(pooled(ids).items()):
        c = pooled_crossings(Ds, tau, ref, threshold)
        if not c:
            continue
        a = Ds[0]["args"]
        t_ref = a["equil_ms"] + tau * (a["a_i"] - ref) / (a["a_i"] - a["a_f"])
        d = median(c)
        during = t_ref + d <= a["equil_ms"] + tau + 1e-8
        rows.append((tau, d, during, a_of_t(a, tau)(t_ref + d)))
    return rows


def plot_onsets(ax, rows, color, marker="o", ls="-", label=None, reverse=False):
    """Line through all points; filled markers for onsets during the ramp, open for
    onsets in the final hold (excluded from the linear-ramp fits)."""
    x = [r[0] for r in rows]
    ax.plot(x, [r[1] for r in rows], ls=ls, color=color, lw=1.0)
    ramp = [r for r in rows if r[2]]
    hold = [r for r in rows if not r[2]]
    ax.plot([r[0] for r in ramp], [r[1] for r in ramp], marker, ls="none", color=color,
            mfc="white" if reverse else color, label=label)
    if hold:
        ax.plot([r[0] for r in hold], [r[1] for r in hold], marker, ls="none", color=color, mfc="white",
                mew=0.9)



W2 = 7.0            # width of a two-column figure (in)
W1 = 3.4            # width of a one-column figure (in)
UM = r"\,\mu\mathrm{m}^{-1}"
DENS = {700: rf"$700{UM}$", 2500: rf"$2500{UM}$", 6250: rf"$6250{UM}$"}


def style():
    """Journal style: figures drawn at their printed size, 8 pt STIX (Times-like) text."""
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "lines.markersize": 3.2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False, "legend.handlelength": 1.6,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.formatter.use_mathtext": True})


def panel(ax, label):
    ax.set_title(label, loc="left", fontsize=8, fontweight="bold", pad=3)


def tq_axis(ax):
    ax.set_xlabel(r"$\tau_Q$ (ms)")


def plain_log(ax, axes="xy"):
    """Log axes labelled with plain numbers (10, 20, 50, ...) instead of 2x10^1."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for name in axes:
        axis = ax.xaxis if name == "x" else ax.yaxis
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())


def fig_freezeout(run):
    """(a) t_hat after the spinodal vs tau_Q, (b) after a*, (c) zeta vs the
    freeze-out threshold from the latest tube_ramp_errors run."""
    fig, (ax, bx, zx) = plt.subplots(1, 3, figsize=(W2, 2.3))
    for n, ids in FORWARD.items():
        for ref, axis in ((SPINODAL[n], ax), (A_STAR[n], bx)):
            plot_onsets(axis, onsets(ids, ref), COLORS[n], label=DENS[n] + (" (cont.)" if n == 2500 else ""))
    for n, ids in REVERSE.items():
        plot_onsets(ax, onsets(ids, BRANCH_END[n]), COLORS[n], marker="s", ls="--",
                    label=DENS[n] + ", reverse", reverse=True)
    ax.plot([], [], "o", color="0.4", mfc="white", label="onset after the ramp")
    for axis in (ax, bx):
        axis.set_xscale("log")
        axis.set_yscale("log")
    tq = torch.tensor([10.0, 400.0])
    ax.loglog(tq, 3.0 * (tq / 10) ** (1 / 3), "k:", lw=0.8)
    ax.text(300, 3.0 * 30 ** (1 / 3) * 0.86, r"$\propto\tau_Q^{1/3}$", fontsize=7, ha="center", va="top")
    tq_axis(ax)
    ax.set_ylabel(r"$\hat t$ after $a_\mathrm{rot}^*$ (ms)")
    bx.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=6)
    plain_log(ax)
    panel(ax, "(a)")
    plain_log(bx)
    tq_axis(bx)
    bx.set_ylabel(r"$\hat t$ after $a^*$ (ms)")
    panel(bx, "(b)")
    err = sorted(ROOT.glob("*_tube_ramp_errors"))[-1] / "errors.json"
    E = json.loads(err.read_text(encoding="utf-8"))
    for name, e in E.items():
        n = int(name.split()[0])
        rev = "reverse" in name
        thr = sorted((float(k) for k in e["thresholds"]), reverse=True)
        zx.errorbar(thr, [e["thresholds"][str(t)]["zeta"] for t in thr],
                    yerr=[e["thresholds"][str(t)]["boot_sd"] for t in thr], marker="s" if rev else "o",
                    capsize=1.5, elinewidth=0.6, color=COLORS[n], ls="--" if rev else "-",
                    mfc="white" if rev else None, label=f"{n}" + (", reverse" if rev else ""))
    zx.axhline(1 / 3, color="k", ls=":", lw=0.8)
    zx.text(0.982, 1 / 3 - 0.0035, r"$1/3$", fontsize=7, va="top")
    zx.invert_xaxis()
    zx.set_xlabel(r"threshold $f_\mathrm{th}$")
    zx.set_ylabel(r"$\zeta$")
    zx.legend(loc="center right", fontsize=6)
    panel(zx, "(c)")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save(fig, run, "fig_freezeout")


def fig_dynamics(run):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
    data = {tau: Ds[0] for tau, Ds in pooled(FORWARD[6250]).items()}
    cmap = plt.get_cmap("viridis")
    taus = sorted(data)
    for k, tau in enumerate(taus):
        D = data[tau]
        a = D["args"]
        t = torch.tensor(D["times_ms"])
        t_rot = a["equil_ms"] + tau * (a["a_i"] - SPINODAL[6250]) / (a["a_i"] - a["a_f"])
        a1.plot(t - t_rot, D["fs"].mean(1), color=cmap(k / (len(taus) - 1) * 0.9),
                label=rf"$\tau_Q = {tau:g}$ ms")
    a1.axvline(0, color="gray", ls=":", lw=0.8)
    a1.set_xlim(-10, 40)
    a1.set_xlabel(r"$t - t(a_\mathrm{rot}^*)$ (ms)")
    a1.set_ylabel(r"$\langle f_s\rangle$")
    a1.legend(loc="lower left", fontsize=6)
    panel(a1, "(a)")
    D = data[100.0]
    l_um = D["dz_um"] / D["args"]["dx"]
    line = D["line"][:, 0] / l_um / 1e3                          # 10^3 atoms per um
    gt = D.get("g2_times_ms", D["times_ms"])
    L_um = D["dz_um"] * line.shape[-1]
    im = a2.imshow(line.T.numpy(), aspect="auto", origin="lower", cmap="magma",
                   extent=(gt[0], gt[-1], 0, L_um), interpolation="nearest", rasterized=True)
    a = D["args"]
    t_rot = a["equil_ms"] + 100 * (a["a_i"] - SPINODAL[6250]) / (a["a_i"] - a["a_f"])
    a2.axvline(t_rot, color="white", ls="--", lw=0.8)
    a2.set_xlabel(r"$t$ (ms)")
    a2.set_ylabel(r"$z$ ($\mu$m)")
    a2.tick_params(top=False, right=False)
    cb = fig.colorbar(im, ax=a2, pad=0.015)
    cb.set_label(r"$n(z)$ ($10^3\,\mu$m$^{-1}$)")
    cb.ax.tick_params(labelsize=7, width=0.5)
    panel(a2, "(b)")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save(fig, run, "fig_dynamics")


def crystal_soft_mode():
    """(a_s, omega) of the soft q = 0 crystal mode from the tube_crystal_bdg runs
    (lowest mode above the gauge mode; without deflation the translation mode,
    ~1e-3, comes first and is skipped)."""
    pts = {}
    for d in sorted(ROOT.glob("*_tube_crystal_bdg")):
        f = d / "crystal_bdg.json"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if not f.exists() or meta.get("status") not in ("finished", "failed"):
            continue                                  # rows are saved per a_s, so a later failure keeps them
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            w = r.get("bands", {}).get("0.0")
            if w:
                pts[r["a_s"]] = next(x for x in w[1:] if x > 0.01)
    return sorted(pts.items())


def fig_hysteresis(run):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.4))
    for ids, ref, mk, lab in ((FORWARD[6250], SPINODAL[6250], "v", "forward"),
                              (REVERSE[6250], BRANCH_END[6250], "^", "reverse")):
        rows = onsets(ids, ref)
        col = COLORS[6250] if "forward" in lab else "#7f6000"
        ax.semilogx([r[0] for r in rows], [r[3] for r in rows], "-", color=col, lw=1.0)
        ramp = [r for r in rows if r[2]]
        hold = [r for r in rows if not r[2]]
        ax.plot([r[0] for r in ramp], [r[3] for r in ramp], mk, color=col,
                mfc="white" if "reverse" in lab else col, label=lab)
        if hold:
            ax.plot([r[0] for r in hold], [r[3] for r in hold], "s", color=col, mfc="white",
                    label=r"forward, onset at $a_f$")
    for x, lab, va in ((BRANCH_END[6250], r"$a_\mathrm{end}$", "bottom"), (A_STAR[6250], r"$a^*$", "top"),
                       (SPINODAL[6250], r"$a_\mathrm{rot}^*$", "top")):
        ax.axhline(x, color="gray", ls=":", lw=0.7)
        ax.text(230, x, lab, fontsize=7, color="0.3", va=va)
    ax.axhspan(SPINODAL[6250], BRANCH_END[6250], color="gold", alpha=0.18, lw=0)
    ax.set_xlim(8, 220)
    tq_axis(ax)
    ax.set_ylabel(r"$a_s$ at the transition ($a_0$)")
    ax.legend(loc="lower right", fontsize=6)
    plain_log(ax, "x")
    panel(ax, "(a)")
    pts = crystal_soft_mode()
    a = torch.tensor([q[0] for q in pts], dtype=torch.float64)
    w = torch.tensor([q[1] for q in pts], dtype=torch.float64)
    bx.plot(a, w, "o-", color="#1f4e79", label=r"$\omega/\omega_\perp$")
    bx.tick_params(right=False)
    cx = bx.twinx()
    cx.tick_params(direction="in", labelsize=7, width=0.6)
    cx.plot(a, w ** 4, "s", color="#c0504d", mfc="white", label=r"$(\omega/\omega_\perp)^4$")
    near = a >= 90.3 - 1e-9                                                    # linear fit of omega^4 near the end
    A = torch.stack([torch.ones_like(a[near]), a[near]], 1)
    c = torch.linalg.lstsq(A, (w[near] ** 4).reshape(-1, 1)).solution.reshape(-1)
    a_end = float(-c[0] / c[1])
    xs = torch.linspace(float(a[near].min()), a_end, 20, dtype=torch.float64)
    cx.plot(xs, c[0] + c[1] * xs, "--", color="#c0504d", lw=0.8, label=rf"linear fit, zero at ${a_end:.2f}\,a_0$")
    cx.set_ylabel(r"$(\omega/\omega_\perp)^4$", color="#c0504d")
    cx.set_ylim(0, None)
    bx.axvline(BRANCH_END[6250], color="gray", ls=":", lw=0.7)
    bx.text(BRANCH_END[6250] - 0.006, 0.05, r"$a_\mathrm{end}$", fontsize=7, color="0.3", ha="right")
    bx.set_xlabel(r"$a_s$ ($a_0$)")
    bx.set_ylabel(r"soft mode $\omega/\omega_\perp$ ($q = 0$)", color="#1f4e79")
    bx.set_ylim(0, None)
    h1, l1 = bx.get_legend_handles_labels()
    h2, l2 = cx.get_legend_handles_labels()
    bx.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=6)
    panel(bx, "(b)")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    save(fig, run, "fig_hysteresis")


def holds():
    """Hold runs at 6250/um entered from 92 a0 with the standard noise cutoff; for
    repeated (a_f, noise, T) the run with the most realizations."""
    best = {}
    for d in sorted(ROOT.glob("*_tube_ramp_3d")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        p = meta.get("params", {})
        if p.get("a_c") == 0 and p.get("density") == 6250 and meta.get("status") == "finished" \
                and p.get("a_i") == 92.0 and p.get("noise_cutoff", 1.0) == 1.0 \
                and p.get("a_f", 1e9) < BRANCH_END[6250]:           # inside the window (90.8 a0: thermal-shift holds)
            try:
                D = next(iter(load_tau_files(d).values()))
            except (FileNotFoundError, StopIteration):
                D = None
            if D is not None:
                key = (p["a_f"], p["noise"], p.get("T_nK") if p["noise"] == "thermal" else None)
                if key not in best or D["peaks"].shape[1] > best[key][1]["peaks"].shape[1]:
                    best[key] = (p, D)
    return list(best.values())


def droplet_curve(D):
    """Mean droplet count: deep droplets (> 1.5 x mean) when recorded, else the
    loose count (> 1.2 x mean, which also picks up thermal density fluctuations)."""
    deep = D.get("deep_peaks")
    return (deep if deep is not None else D["peaks"]).float().mean(1), deep is not None


def survival_sets():
    """Pooled deep-count hold data (from 92 a0, standard cutoff) for the survival
    curves: {(a_s, T): (sorted waiting times, n, t_end, L_um, Gamma per um per s)}."""
    import tube_ramp_nucleation as N
    pooled_t1 = {}
    for name, p, D in N.holds():
        if D.get("deep_peaks") is None or p["a_i"] != 92.0 or p.get("noise_cutoff", 1.0) != 1.0:
            continue
        t1, t_end, _ = N.first_times(p, D)
        e = pooled_t1.setdefault((p["a_f"], p["T_nK"]), {"t1": [], "t_end": t_end, "L": p["cells"] * p["cell_um"]})
        e["t1"] += t1
        e["t_end"] = min(e["t_end"], t_end)
    out = {}
    for key, e in pooled_t1.items():
        r = N.estimate(e["t1"], e["t_end"], e["L"])
        out[key] = (sorted(x for x in e["t1"] if x is not None), len(e["t1"]), e["t_end"], e["L"],
                    r["Gamma_per_um_s"])
    return out


def fig_nucleation(run):
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.3))
    (a1, a2), (a3, a4) = axs
    H = holds()
    cols = {89.95: "#1f4e79", 90.05: "#c0504d", 90.15: "#9bbb59", 90.25: "#7f6000"}
    for p, D in sorted(H, key=lambda h: h[0]["a_f"]):
        if p["noise"] == "thermal" and p["T_nK"] == 5.0:
            y, deep = droplet_curve(D)
            a1.plot(D["times_ms"], y, "-" if deep else "--", color=cols.get(p["a_f"], "k"),
                    label=rf"${p['a_f']:.2f}\,a_0$" + ("" if deep else r" ($>1.2\bar n$)"))
    tcol = {0: "0.5", 2.5: "#1f4e79", 5.0: "#c0504d", 10.0: "#e8a33d"}
    for p, D in sorted(H, key=lambda h: (h[0]["noise"] == "thermal", h[0].get("T_nK", 0))):
        if abs(p["a_f"] - 90.05) < 1e-6:
            T = 0 if p["noise"] == "quantum" else p["T_nK"]
            y, deep = droplet_curve(D)
            a2.plot(D["times_ms"], y, "-" if deep else "--", color=tcol.get(T, "k"),
                    label=(r"$T = 0$ (quantum noise)" if T == 0 else rf"$T = {T:g}$ nK"))
    for a, lab, title in ((a1, "(a)", r"$T = 5$ nK"), (a2, "(b)", r"$a_s = 90.05\,a_0$")):
        a.axvline(30, color="gray", ls=":", lw=0.7)
        a.set_xlabel(r"$t$ (ms)")
        a.set_ylabel(r"droplets ($>1.5\,\bar n$)")
        a.set_xlim(0, 182)
        a.legend(loc="upper left", fontsize=6, title=title, title_fontsize=6.5, alignment="left")
        panel(a, lab)
    S = survival_sets()
    for ax, sel, lab, colmap in ((a3, lambda k: k[1] == 5.0, lambda k: rf"${k[0]:.2f}\,a_0$", cols),
                                 (a4, lambda k: k[0] == 90.05, lambda k: rf"${k[1]:g}$ nK", None)):
        for key in sorted(k for k in S if sel(k)):
            ts, n, t_end, L, gamma = S[key]
            color = colmap.get(key[0]) if colmap else tcol.get(key[1])
            ax.step([0] + ts + [t_end], [1] + [1 - (i + 1) / n for i in range(len(ts))] + [1 - len(ts) / n],
                    where="post", color=color, label=lab(key) + f" ($n={n}$)")
            tt = torch.linspace(0, t_end, 100)
            ax.plot(tt, torch.exp(-gamma * L * 1e-3 * tt), ":", color=color, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.03, 1.15)
        ax.set_xlabel(r"$t - t(a^*)$ (ms)")
        ax.set_ylabel(r"$S(t)$")
        ax.legend(loc="lower left", fontsize=6)
    panel(a3, "(c)")
    panel(a4, "(d)")
    a3.text(0.98, 0.9, r"5 nK", transform=a3.transAxes, ha="right", fontsize=7)
    a4.text(0.98, 0.9, r"$90.05\,a_0$", transform=a4.transAxes, ha="right", fontsize=7)
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=0.8)
    save(fig, run, "fig_nucleation")


def fig_droplets(run):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True)
    rows = []
    for tau, Ds in sorted(pooled(FORWARD[700]).items()):
        c = pooled_crossings(Ds, tau, SPINODAL[700])
        final = torch.cat([D["peaks"][-1].float() for D in Ds]).mean()
        rows.append((tau, float(final), median(c), c[0], c[-1]))
    a1.semilogx([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[700])
    a1.axhline(32, color="gray", ls=":", lw=0.7)
    a1.text(150, 32.15, "32 ground-state cells", fontsize=6.5, color="0.3", ha="center")
    a1.set_ylabel("final droplets")
    panel(a1, "(a)")
    flags = {r[0]: r[2] for r in onsets(FORWARD[700], SPINODAL[700])}
    a2.loglog([r[0] for r in rows], [r[2] for r in rows], "-", color=COLORS[700])
    a2.plot([r[0] for r in rows if flags[r[0]]], [r[2] for r in rows if flags[r[0]]], "o", color=COLORS[700],
            label="median")
    a2.plot([r[0] for r in rows if not flags[r[0]]], [r[2] for r in rows if not flags[r[0]]], "o",
            color=COLORS[700], mfc="white", mew=0.9, label="median, onset after the ramp")
    a2.fill_between([r[0] for r in rows], [r[3] for r in rows], [r[4] for r in rows], color=COLORS[700],
                    alpha=0.2, lw=0, label="range over realizations")
    tq_axis(a2)
    a2.set_ylabel(r"$\hat t$ after $a_\mathrm{rot}^*$ (ms)")
    a2.legend(loc="upper left", fontsize=6)
    plain_log(a2)
    panel(a2, "(b)")
    fig.tight_layout(pad=0.4, h_pad=0.6)
    save(fig, run, "fig_droplets")


def latest_states():
    d = sorted(ROOT.glob("*_tube_states"))[-1]
    return load_states(d), d.name


def fig_schematic(run):
    """(a) E_crystal - E_uniform per atom vs a_s at 6250/um (static ground states,
    cell 4.3 l, run 20261004-021532), with the spinodals and a*; (b) protocol;
    (c) the two routes in a schematic Landau energy F(C) = r C^2 - C^3 + C^4 of the
    crystal amplitude C: r = 0 is the uniform spinodal, r = 1/4 is a* and
    r = 9/32 the end of the crystal branch (r plays the role of a_s - a_rot*);
    (d) column densities of the three kinds of state (tube_states.py)."""
    D = json.loads(next(ROOT.glob("20261004-021532_*")).joinpath("tube_first_order_6250_refine2.json")
                   .read_text(encoding="utf-8"))
    cell = min(D["cells"], key=lambda c: abs(c["L"] - 4.3))
    rows = sorted(cell["rows"], key=lambda r: r["a_s"])
    crys = [(r["a_s"], r["dE"]) for r in rows if r.get("C_line", 1) > 0.05]
    fig = plt.figure(figsize=(W2, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.42)
    a1, a2, a3 = (fig.add_subplot(gs[0, k]) for k in range(3))
    a1.plot([c[0] for c in crys], [1e3 * c[1] for c in crys], "o-", color=COLORS[6250], ms=2.5, label="crystal")
    a1.axhline(0, color="k", lw=0.8, label="uniform superfluid")
    a1.axvspan(SPINODAL[6250], BRANCH_END[6250], color="gold", alpha=0.22, lw=0)
    y_top = 1.9
    for x, lab in ((SPINODAL[6250], r"$a_\mathrm{rot}^*$"), (A_STAR[6250], r"$a^*$"),
                   (BRANCH_END[6250], r"$a_\mathrm{end}$")):
        a1.axvline(x, color="gray", ls=":", lw=0.7)
        a1.text(x, y_top, lab, fontsize=7, ha="center", va="bottom")
    a1.set_ylim(-9.5, y_top)
    a1.set_xlabel(r"$a_s$ ($a_0$)")
    a1.set_ylabel(r"$(E_\mathrm{cr} - E_\mathrm{u})/N$ ($10^{-3}\hbar\omega_\perp$)")
    a1.legend(loc="lower right", fontsize=6)
    panel(a1, "(a)")
    t = [0, 10, 60, 70]
    a2.plot(t, [92, 92, 88, 88], "-", color="#1f4e79", label="forward")
    a2.plot(t, [88.5, 88.5, 92, 92], "--", color="#7f6000", label="reverse")
    a2.plot([0, 10, 30, 110], [92, 92, 90.05, 90.05], ":", color="#c0504d", label="hold")
    a2.axhspan(SPINODAL[6250], BRANCH_END[6250], color="gold", alpha=0.22, lw=0)
    a2.set_xlim(0, 110)
    a2.set_xlabel(r"$t$ (ms)")
    a2.set_ylabel(r"$a_s(t)$ ($a_0$)")
    a2.legend(loc="lower right", fontsize=6)
    panel(a2, "(b)")
    C = torch.linspace(0, 0.85, 300)
    curves = [(0.27, r"$a_s > a^*$", "0.55", "--"), (0.25, r"$a_s = a^*$", "k", "-"),
              (0.2, r"$a_\mathrm{rot}^* < a_s < a^*$", "#1f4e79", "-"), (0.0, r"$a_s = a_\mathrm{rot}^*$", "#c0504d", "-")]
    for r, lab, col, ls in curves:
        a3.plot(C, 1e2 * (r * C ** 2 - C ** 3 + C ** 4), ls=ls, color=col, label=lab)
    r = 0.2
    cb = (3 - (9 - 32 * r) ** 0.5) / 8
    F = lambda c: 1e2 * (r * c ** 2 - c ** 3 + c ** 4)  # noqa: E731
    a3.plot([0], [0], "o", color="#1f4e79", ms=3.5)
    a3.annotate("", xy=(cb + 0.12, F(cb) + 0.05), xytext=(0.02, 0.12),
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8, connectionstyle="arc3,rad=-0.35"))
    a3.text(0.03, 0.55, "1: nucleation", fontsize=6.5, color="#1f4e79")
    a3.annotate("", xy=(0.27, 1e2 * (-0.27 ** 3 + 0.27 ** 4)), xytext=(0.04, -0.15),
                arrowprops=dict(arrowstyle="->", color="#c0504d", lw=0.8))
    a3.text(0.33, -2.05, "2: spinodal", fontsize=6.5, color="#c0504d")
    a3.set_ylim(-2.3, 1.1)
    a3.set_xlim(0, 1.5)
    a3.set_xlabel(r"crystal amplitude $C$")
    a3.set_ylabel(r"$F(C)$ (arb. units)")
    a3.legend(loc="upper right", fontsize=5.8, handlelength=1.3)
    panel(a3, "(c)")
    states, _ = latest_states()
    titles = {"uniform": "uniform superfluid", "supersolid": "supersolid", "droplets": "isolated droplets"}
    vmax = {6250: float(states["supersolid"]["column"].max()), 700: float(states["droplets"]["column"].max())}
    for k, name in enumerate(("uniform", "supersolid", "droplets")):
        st = states[name]
        ax = fig.add_subplot(gs[1, k])
        col = torch.cat([st["column"]] * 3, dim=1)                   # three cells along z
        l_um = st["l_um"]
        y = st["y"] * l_um
        Lz = 3 * st["cell_um"]
        ny = (y.abs() <= 6.0)
        im = col[ny].numpy() / vmax[int(st["density"])]           # same scale for the two 6250/um states
        ax.imshow(im, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                  extent=(0, Lz, float(y[ny][0]), float(y[ny][-1])), interpolation="bilinear", rasterized=True)
        ax.tick_params(top=False, right=False, colors="k")
        ax.set_xlabel(r"$z$ ($\mu$m)")
        if k == 0:
            ax.set_ylabel(r"$y$ ($\mu$m)")
        ax.set_title(rf"{titles[name]}: ${int(st['density'])}{UM}$, ${st['a_s']:g}\,a_0$", fontsize=7, pad=2)
        if k == 0:
            ax.text(-0.33, 1.13, "(d)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.08)
    save(fig, run, "fig_schematic")


COEX_RUN = "20261008-123945"                    # tube_coexistence.py, 6250/um, a_s = 90.15, 90.256, 90.35
LOCALIZED_RUN = "20261008-130517"               # tube_localized.py, 32 cells (88 um), 90.25 a0


def static_file(run_id, name):
    """A file of a static run record in runs/, or its copy in the reduced archive
    (data_C1/static), which also holds the .npz files that git does not track."""
    found = [d / name for d in sorted(ROOT.glob(run_id + "_*")) if (d / name).exists()]
    if found and not __import__("os").environ.get("C1_ARCHIVE_ONLY"):
        return found[0]
    return next((Path(__file__).resolve().parents[1] / "data_C1" / "static").glob(run_id + "_*")) / name


def fig_coexistence(run):
    """(a) Double tangent at a* (6250/um): energy per length of the uniform and
    crystal branches minus the common tangent, which touches them at the
    coexisting densities n_M and n_U; (b) coexistence window in the (a_s, n)
    plane with the C1 holds at 6250/um; (c) column density and (d) one-cell
    running mean of the line density of the localized state in the 88 um tube
    at 90.25 a0."""
    import numpy as np
    D = json.loads(static_file(COEX_RUN, "tube_coexistence.json").read_text(encoding="utf-8"))
    Lc = json.loads(static_file(LOCALIZED_RUN, "tube_localized.json").read_text(encoding="utf-8"))
    col = np.load(static_file(LOCALIZED_RUN, "localized_column.npz"))
    l_um = 0.6409970693791566                        # oscillator length (um) for omega_perp = 2 pi x 150 Hz
    fig = plt.figure(figsize=(W2, 4.1))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 0.62, 0.75], hspace=0.62, wspace=0.3)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    e = min(D["per_a"], key=lambda r: abs(r["a_s"] - 90.256))
    co = e["coexistence"]
    mu = co["mu"]
    to_l = lambda n: n * l_um                         # noqa: E731  atoms per l from atoms per um

    def branch(rows):
        n = np.array([r["n"] for r in rows])
        return n, to_l(n) * np.array([r["e"] for r in rows])          # energy per l (hbar omega)
    nU, fU = branch(e["uniform"])
    nM, fM = branch(e["crystal_opt"])
    cU, cM = np.polyfit(nU - 6250, fU, 4), np.polyfit(nM - 6250, fM, 4)
    P = mu * to_l(co["n_U"]) - np.polyval(cU, co["n_U"] - 6250)      # common tangent f = mu n - P
    g = lambda c, n: (np.polyval(c, n - 6250) - (mu * to_l(n) - P)) / l_um   # noqa: E731  per um
    x = np.linspace(6175, 6325, 400)
    a1.plot(x, g(cU, x), "k-", label="uniform superfluid")
    a1.plot(x, g(cM, x), "-", color=COLORS[6250], label="crystal")
    su, sm = (nU >= 6175) & (nU <= 6325), (nM >= 6175) & (nM <= 6325)
    a1.plot(nU[su], g(cU, nU[su]), "ko", ms=2.2)
    a1.plot(nM[sm], g(cM, nM[sm]), "o", color=COLORS[6250], ms=2.2)
    a1.axhline(0, color="gray", lw=0.7, ls="--")
    a1.axvspan(co["n_M"], co["n_U"], color="gold", alpha=0.25, lw=0)
    for n, lab in ((co["n_M"], r"$n_M$"), (co["n_U"], r"$n_U$")):
        a1.plot([n], [0], "o", mfc="white", mec="k", ms=3.2, zorder=5)
        a1.text(n, -0.32, lab, ha="center", va="top", fontsize=7)
    a1.set_xlim(6175, 6325)
    a1.set_ylim(-0.75, 3.0)
    a1.set_xlabel(r"line density $n$ ($\mu$m$^{-1}$)")
    a1.set_ylabel(r"$f - (\mu^* n - P^*)$ ($\hbar\omega_\perp/\mu$m)")
    a1.legend(loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.0))
    a1.text(0.98, 0.05, rf"$a_s = {e['a_s']:g}\,a_0$", transform=a1.transAxes, fontsize=7, ha="right")
    panel(a1, "(a)")
    pts = sorted((r["a_s"], r["coexistence"]["n_M"], r["coexistence"]["n_U"]) for r in D["per_a"])
    av = np.array([p[0] for p in pts])
    pM, pU = np.polyfit(av, [p[1] for p in pts], 2), np.polyfit(av, [p[2] for p in pts], 2)
    xa = np.linspace(90.135, 90.365, 200)
    a2.fill_between(xa, np.polyval(pM, xa), np.polyval(pU, xa), color="gold", alpha=0.35, lw=0, label="coexistence")
    a2.plot(xa, np.polyval(pM, xa), "-", color=COLORS[6250], label=r"$n_M$ (crystal)")
    a2.plot(xa, np.polyval(pU, xa), "k-", label=r"$n_U$ (uniform)")
    a2.plot(av, [p[1] for p in pts], "o", color=COLORS[6250], ms=2.5)
    a2.plot(av, [p[2] for p in pts], "ko", ms=2.5)
    a2.axhline(6250, color="gray", lw=0.7, ls="--")
    w = D["window"]
    a2.plot([w["a_from_crystal_side"], w["a_from_uniform_side"]], [6250, 6250], "-", color="#c0504d", lw=2.2,
            solid_capstyle="butt")
    for a_h in (90.15, 90.25):
        a2.plot([a_h], [6250], "v", color="#1f4e79", ms=3.5)
    a2.axvline(A_STAR[6250], color="gray", ls=":", lw=0.7)
    a2.text(A_STAR[6250] + 0.004, 6405, r"$a^*$", fontsize=7)
    a2.text(90.153, 6262, "holds", fontsize=6.5, color="#1f4e79")
    a2.set_xlim(90.135, 90.365)
    a2.set_xlabel(r"$a_s$ ($a_0$)")
    a2.set_ylabel(r"$n$ ($\mu$m$^{-1}$)")
    a2.legend(loc="upper right", fontsize=6)
    panel(a2, "(b)")
    a3 = fig.add_subplot(gs[1, :])
    c = col["column_yz"]
    dz, dy = float(col["dz_um"]), float(col["dy_um"])
    Ny, Nz = c.shape
    y = (np.arange(Ny) - Ny / 2) * dy
    keep = np.abs(y) <= 5.0
    a3.imshow(c[keep], origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=float(c.max()),
              extent=(0, Nz * dz, float(y[keep][0]), float(y[keep][-1])), interpolation="bilinear", rasterized=True)
    a3.tick_params(top=False, right=False, labelbottom=False)
    a3.set_ylabel(r"$y$ ($\mu$m)")
    best = min(Lc["states"], key=lambda s: s["e"])
    a3.set_title(rf"localized state: ${Lc['density']:g}{UM}$, ${Lc['a_s']:g}\,a_0$, {Lc['cells']} cells "
                 rf"({Lc['tube_um']:.0f} $\mu$m)", fontsize=7, pad=2)
    a3.text(-0.075, 1.12, "(c)", transform=a3.transAxes, fontsize=8, fontweight="bold")
    a4 = fig.add_subplot(gs[2, :], sharex=a3)
    line = np.array(best["line"]) / l_um                              # atoms per um
    k = 2 * np.pi * np.fft.fftfreq(line.size, d=dz)
    k_cell = 2 * np.pi / (Lc["tube_um"] / Lc["cells"])
    smooth = np.fft.ifft(np.fft.fft(line) * np.exp(-0.5 * (k / (0.25 * k_cell)) ** 2)).real   # local mean
    zz = np.arange(line.size) * dz
    a4.plot(zz, smooth, color=COLORS[6250], lw=1.0)
    for n, ls, lab in ((Lc["n_M"], ":", r"$n_M$"), (Lc["n_U"], "--", r"$n_U$")):
        a4.axhline(n, color="k", lw=0.7, ls=ls)
    a4.text(zz[-1] * 0.5, Lc["n_M"] - 12, r"$n_M$ (coexistence)", ha="center", fontsize=6.5)
    a4.text(zz[-1] * 0.99, Lc["n_U"] + 4, r"$n_U$", ha="right", fontsize=6.5)
    a4.set_ylim(Lc["n_M"] - 30, Lc["n_U"] + 25)
    a4.set_ylabel(r"local mean ($\mu$m$^{-1}$)", fontsize=7)
    a4.set_xlabel(r"$z$ ($\mu$m)")
    a4.set_xlim(0, Nz * dz)
    a4.text(-0.075, 1.08, "(d)", transform=a4.transAxes, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
    save(fig, run, "fig_coexistence")


def main():
    style()
    with runs.Run("tube_ramp_figures", {"forward": FORWARD, "reverse": REVERSE, "spinodal": SPINODAL,
                                        "a_star": A_STAR}) as run:
        for f in (fig_schematic, fig_freezeout, fig_dynamics, fig_hysteresis, fig_nucleation, fig_droplets,
                  fig_coexistence):
            f(run)
            print(f.__name__, "done", flush=True)
        run.result(figures=["fig_schematic.png", "fig_freezeout.png", "fig_dynamics.png", "fig_hysteresis.png", "fig_nucleation.png",
                            "fig_droplets.png", "fig_coexistence.png"])


if __name__ == "__main__":
    main()

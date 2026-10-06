"""Kibble-Zurek analysis of tube_ramp.py runs: freeze-out time and correlation
length versus the ramp time tau_Q.

- Freeze-out t_hat: first time after the critical crossing t_c at which the
  sample mean <f_s> drops below 0.98 (Kirkby et al. 2025).
- Correlation length X: the sample-averaged g2(dz) at t_c + t_hat fitted with
  A + (1 - A) cos(K dz) exp(-dz^2 / X^2) (their Eq. (8)), dz up to --fit-um.
- Power laws t_hat ~ tau_Q^zeta, X ~ tau_Q^nu_KZ by least squares in log-log.
Reference (n = 2500, quantum noise): zeta = 0.346(2); thermal 0.352(3), X 0.335(3).

  uv run python benchmarks/tube_ramp_analysis.py RUN [RUN ...] [--fit-um 60]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dipgpe import runs  # noqa: E402


def power_law(x, y):
    lx, ly = torch.log(torch.tensor(x, dtype=torch.float64)), torch.log(torch.tensor(y, dtype=torch.float64))
    A = torch.stack([lx, torch.ones_like(lx)], 1)
    sol = torch.linalg.lstsq(A, ly[:, None]).solution[:, 0]
    resid = ly - A @ sol
    n = len(x)
    if n > 2:
        s2 = float((resid ** 2).sum()) / (n - 2)
        cov = s2 * torch.linalg.inv(A.T @ A)
        err = math.sqrt(float(cov[0, 0]))
    else:
        err = float("nan")
    return float(sol[0]), err


def fit_envelope(dz, g2, K0, steps=4000):
    """Least squares for A, K, X in A + (1 - A) cos(K dz) exp(-dz^2/X^2) (Adam on log X)."""
    A = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    K = torch.tensor(K0, dtype=torch.float64, requires_grad=True)
    lX = torch.tensor(math.log(float(dz[-1]) / 3), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([A, K, lX], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        model = A + (1 - A) * torch.cos(K * dz) * torch.exp(-(dz / torch.exp(lX)) ** 2)
        loss = ((model - g2) ** 2).mean()
        loss.backward()
        opt.step()
    return float(torch.exp(lX).detach()), float(K.detach()), float(A.detach()), float(loss.detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--fit-um", type=float, default=60.0)
    p.add_argument("--a-c", type=float, default=None, help="override the critical a_s used for t_c")
    p.add_argument("--estimator", choices=["mean", "median"], default="mean",
                   help="mean: first time <f_s> crosses 0.98 (Kirkby et al.); median: median over realizations "
                        "of each realization's crossing (robust to single early realizations)")
    p.add_argument("--pool", action="store_true",
                   help="pool realizations of equal tau_Q across runs (same protocol and records)")
    args = p.parse_args()
    rows = []
    entries = []
    for ref in args.runs:
        d = Path(runs.resolve(ref))
        for f in sorted((d / "data").glob("tau_*ms.pt")):
            entries.append((d, float(f.stem[4:-2]), torch.load(f)))
    if args.pool:                       # concatenate f_s over realizations; g2 averaged by realization count
        pooled = {}
        for d, tau, D in entries:
            if tau not in pooled:
                pooled[tau] = (d, tau, dict(D))
            else:
                P = pooled[tau][2]
                n0, n1 = P["fs"].shape[1], D["fs"].shape[1]
                assert len(P["times_ms"]) == len(D["times_ms"]), "pooled runs need the same records"
                P["g2"] = (P["g2"] * n0 + D["g2"] * n1) / (n0 + n1)
                P["fs"] = torch.cat([P["fs"], D["fs"]], 1)
        entries = list(pooled.values())
    for d, tau, D in entries:
        a = D["args"]
        times = torch.tensor(D["times_ms"])
        mean_fs = D["fs"].mean(1)
        a_c = args.a_c if args.a_c is not None else a["a_c"]
        t_c = a["equil_ms"] + tau * (a["a_i"] - a_c) / (a["a_i"] - a["a_f"])
        melt = a["a_f"] > a["a_i"]                             # reverse ramp: <f_s> rises above 0.98
        idx = [i for i in range(len(times)) if times[i] >= t_c and
               (mean_fs[i] > 0.98 if melt else mean_fs[i] < 0.98)]
        if args.estimator == "median" and idx:
            per = []
            for s_ in range(D["fs"].shape[1]):
                f_ = D["fs"][:, s_]
                k = next((i for i in range(len(times)) if times[i] >= t_c and
                          (f_[i] > 0.98 if melt else f_[i] < 0.98)), None)
                if k is not None:
                    per.append(k)
            idx = [sorted(per)[len(per) // 2]] if per else []
        if not idx:
            print(f"{d.name} tau {tau:g}: no freeze-out")
            continue
        i = idx[0]
        g2_times = D.get("g2_times_ms", D["times_ms"])          # g2 may be stored less often
        j = min(range(len(g2_times)), key=lambda q: abs(g2_times[q] - float(times[i])))
        g2 = D["g2"][j].to(torch.float64)
        dz = torch.arange(len(g2), dtype=torch.float64) * D["dz_um"]
        m = dz <= args.fit_um
        cell = a["cell_um"]
        X, K, A, loss = fit_envelope(dz[m], g2[m] / g2[0], 2 * math.pi / cell)   # normalized: 1 at dz = 0
        rows.append((tau, float(times[i]) - t_c, X))
        print(f"{d.name} tau_Q = {tau:g} ms: t_hat = {float(times[i]) - t_c:.3f} ms, X = {X:.2f} um "
              f"(K = {K:.3f}/um, period {2 * math.pi / K:.2f} um, A = {A:.3f}, residual {loss:.1e}), "
              f"samples {D['fs'].shape[1]}")
    rows.sort()
    if len(rows) >= 2:
        z, ze = power_law([r[0] for r in rows], [r[1] for r in rows])
        n, ne = power_law([r[0] for r in rows], [r[2] for r in rows])
        print(f"freeze-out exponent zeta = {z:.3f} +- {ze:.3f}; correlation-length exponent {n:.3f} +- {ne:.3f} "
              f"({len(rows)} ramp times)")


if __name__ == "__main__":
    main()

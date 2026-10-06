"""Time-step convergence of the freeze-out (paper C1): the production forward
ramps at 6250/um (dt = 10 us, --dt-safety 0.9) against the same ramps with half
the time step (--dt-safety 0.45). Both use seed 0, so the initial noise is
identical and the onsets can be compared realization by realization.

  uv run python benchmarks/tube_ramp_dt_check.py <halved-dt run id>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dipgpe import runs  # noqa: E402
import tube_ramp_errors as E  # noqa: E402
import tube_ramp_figures as F  # noqa: E402


def main():
    fine = F.load(sys.argv[1])
    base = F.pooled(F.FORWARD[6250])
    out = {}
    with runs.Run("tube_ramp_dt_check", {"fine": sys.argv[1], "base": F.FORWARD[6250]}) as run:
        for tau, Df in sorted(fine.items()):
            Db = base[tau][0]
            assert Db["args"]["seed"] == Df["args"]["seed"]
            res = {}
            for thr in (0.98, 0.9, 0.7):
                cb = E.per_realization(Db, tau, F.SPINODAL[6250], thr)
                cf = E.per_realization(Df, tau, F.SPINODAL[6250], thr)
                pairs = [(b, f) for b, f in zip(cb, cf) if b is not None and f is not None]
                diffs = [f - b for b, f in pairs]
                mb = E.median([b for b, _ in pairs])
                mf = E.median([f for _, f in pairs])
                res[str(thr)] = {"median_base_ms": mb, "median_fine_ms": mf, "median_change_ms": mf - mb,
                                 "relative": (mf - mb) / mb, "max_abs_pair_diff_ms": max(abs(d) for d in diffs),
                                 "pairs": len(pairs)}
                print(f"tau_Q {tau:g} ms, f_s < {thr}: median t_hat {mb:.3f} -> {mf:.3f} ms "
                      f"({100 * (mf - mb) / mb:+.2f}%), largest per-realization change "
                      f"{max(abs(d) for d in diffs):.2f} ms ({len(pairs)} pairs)", flush=True)
            out[str(tau)] = res
        run.save_json("dt_check.json", out)


if __name__ == "__main__":
    main()

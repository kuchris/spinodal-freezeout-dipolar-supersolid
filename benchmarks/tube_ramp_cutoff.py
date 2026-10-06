"""Noise-cutoff sensitivity (paper C1): freeze-out with quantum noise and the
nucleation rate with thermal noise for the noise cutoff scaled by 0.7 and 1.3
(tube_ramp.py --noise-cutoff; quantum: k_max = 1/xi, thermal: 2 kT).

Quantum: median freeze-out time t_hat (f_s < 0.98 after the spinodal) at
tau_Q = 20 and 100 ms and the two-point exponent ln(t100/t20)/ln 5, against the
production runs (cutoff 1). Thermal: Gamma at 90.05 a0, 5 nK, from the holds
(censored-exponential MLE as in tube_ramp_nucleation.py).

  uv run python benchmarks/tube_ramp_cutoff.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dipgpe import runs  # noqa: E402
import tube_ramp_errors as E  # noqa: E402
import tube_ramp_figures as F  # noqa: E402
import tube_ramp_nucleation as N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs"


def quantum_sets():
    """{cutoff: {tau: [D, ...]}} for 6250/um forward quantum ramps 92 -> 88 a0."""
    sets = {1.0: E.dataset(F.FORWARD[6250])}
    for d in sorted(ROOT.glob("*_tube_ramp_3d")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        p = meta.get("params", {})
        c = p.get("noise_cutoff", 1.0)
        if c != 1.0 and p.get("noise") == "quantum" and p.get("density") == 6250 and p.get("a_f") == 88.0 \
                and meta.get("status") == "finished":
            for tau, D in F.load(d.name.split("_")[0]).items():
                sets.setdefault(c, {}).setdefault(tau, []).append(D)
    return sets


def main():
    with runs.Run("tube_ramp_cutoff", {}) as run:
        out = {"quantum": {}, "thermal": []}
        ref = F.SPINODAL[6250]
        for c, data in sorted(quantum_sets().items()):
            med = {}
            for tau in (20.0, 100.0):
                vals = [x for D in data.get(tau, []) for x in E.per_realization(D, tau, ref, 0.98)
                        if x is not None and x > 0]
                med[tau] = E.median(vals) if vals else None
            z = (math.log(med[100.0] / med[20.0]) / math.log(5)) if med[20.0] and med[100.0] else None
            out["quantum"][str(c)] = {"t_hat_20": med[20.0], "t_hat_100": med[100.0], "zeta_2pt": z}
            print(f"quantum cutoff x{c:g}: t_hat = {med[20.0]} ms (tau_Q 20), {med[100.0]} ms (100); "
                  f"two-point zeta {z if z is None else round(z, 3)}", flush=True)
        for name, p, D in N.holds():
            if p["a_f"] == 90.05 and p["T_nK"] == 5.0 and p["a_i"] == 92.0 and D.get("deep_peaks") is not None:
                t1, t_end, _ = N.first_times(p, D)
                r = N.estimate(t1, t_end, p["cells"] * p["cell_um"])
                r.update(run=name, noise_cutoff=p.get("noise_cutoff", 1.0),
                         S_rot_at_hold_start=N.roton_power(D, p["equil_ms"] + 20.0))
                out["thermal"].append(r)
                print(f"thermal cutoff x{r['noise_cutoff']:g} ({name}): {r['nucleated']}/{r['realizations']}, "
                      f"Gamma = {r['Gamma_per_um_s']:.3g} +- {r['Gamma_err']:.2g} per um per s, "
                      f"roton power at the hold start {r['S_rot_at_hold_start']:.3f}", flush=True)
        run.save_json("cutoff.json", out)


if __name__ == "__main__":
    main()

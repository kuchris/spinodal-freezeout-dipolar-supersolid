"""Export the reduced data set of paper C1 for a public archive (Zenodo).

Every finished 3D tube-ramp run that docs/RESULTS.md cites is written as
compressed NumPy files with the per-realization observables (no wavefunctions):

  data_C1/<run id>/params.json       command-line parameters, code version, environment
  data_C1/<run id>/tau_<t>ms.npz     arrays (see data_C1/README.md)
  data_C1/static/                    ground states and Bogoliubov results used in the paper
  data_C1/index.csv                  one row per run and ramp time, with its role in the paper
  data_C1/MANIFEST.sha256            checksums
  data_C1.zip                        the whole folder

  uv run python tools/export_c1_data.py
"""

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import tube_ramp_figures as F  # noqa: E402

OUT = ROOT / "data_C1"
FIELDS = {"times_ms": np.float32, "fs": np.float32, "contrast": np.float32, "peaks": np.int16,
          "deep_peaks": np.int16, "fs_avg_1ms": np.float32, "g2": np.float32, "g2_times_ms": np.float32,
          "line": np.float32}
STATIC = ["20261004-021532", "20261005-143507", "20261005-144208", "20261005-144924", "20261005-164310",
          "20261005-185459", "20261006-002643", "20261006-005533", "20261006-010515", "20261006-011016",
          "20261006-133351",
          # v2, Appendix E: Maxwell construction (6250/um, 2500/um, convergence checks) and localized states
          "20261008-123945", "20261008-123613", "20261008-154519", "20261008-154742", "20261008-154909",
          "20261008-130517", "20261008-131726", "20261008-132115", "20261008-132458"]
V2_ROLES = {"20261008-133754": "hold at 90.8 a0, 5 nK: roton frequencies (Appendix F)",
            "20261008-135021": "hold at 90.8 a0, 10 nK: roton frequencies (Appendix F)",
            "20261008-141000": "hold at 90.8 a0, quantum noise: zero-temperature reference (Appendix F)",
            "20261008-140146": "thermal ramps, 5 nK, tau_Q 20 and 200 ms (Table V)"}

README = """# Reduced data for "Spinodal-controlled freeze-out and nucleation at the
first-order superfluid--supersolid transition of a dipolar gas" (W. K. Wong)

Per-realization observables of the truncated-Wigner simulations (no
wavefunctions). Every run directory holds `params.json` (the full parameter set,
command, git commit and environment of the run) and one `tau_<t>ms.npz` per
ramp time. `index.csv` lists all files with density, protocol, noise and the
role of the run in the paper.

Arrays in `tau_<t>ms.npz` (R = realizations, K = records, every `record_ms`):

| name | shape | meaning |
|---|---|---|
| times_ms | (K,) | time since the start of the run (ms); the ramp starts at `equil_ms` |
| fs | (K, R) | Leggett superfluid fraction of the line density, L^2 / (N int dz / n(z)) |
| contrast | (K, R) | line-density contrast (max - min) / (max + min) |
| peaks | (K, R) | local maxima of n(z) above 1.2 x mean (droplet count) |
| deep_peaks | (K, R) | local maxima above 1.5 x mean (later runs only) |
| fs_avg_1ms | (K, R) | f_s of the line density averaged over 1 ms (later runs only) |
| g2 | (K2, Nz/2) | sample-averaged density correlation g2(dz), every `store_every` records |
| g2_times_ms | (K2,) | times of g2 and line |
| line | (K2, S, Nz) | line density n(z) (atoms per oscillator length l = 0.641 um) of the first S realizations |
| dz_um | () | grid spacing along the tube in um |

The scattering length follows a_s(t) = a_i + (a_f - a_i) min(1, max(0, (t - equil_ms)/tau)),
with a_i, a_f, equil_ms in params.json (a_s in Bohr radii). Spinodals and
transition points: a_rot* = 89.845, 92.314, 84.506 a0 and a* = 90.256, 92.314,
85.28 a0 at 6250, 2500, 700 per um; end of the crystal branch 90.39 a0 at 6250 per um.

`static/` holds the ground-state energies (tube_first_order), the Bogoliubov
roton and crystal soft-mode results and the ground states of Fig. 1(d), each with
its run metadata; and, for Appendix E, the Maxwell construction of the uniform
and crystal branches (tube_coexistence.json: energies, chemical potentials and
pressures per density, coexisting densities and the coexistence window) and the
localized states (tube_localized.json with the line densities of the converged
states; localized_column.npz with the column density of Fig. 7).

Analysis and figures: the scripts benchmarks/tube_ramp_errors.py,
tube_ramp_figures.py, tube_ramp_nucleation.py, tube_ramp_cutoff.py,
tube_ramp_dt_check.py and tube_thermal_shift.py of the dipgpe code read the
original run records or, without them, the arrays here (the same observables in
a portable format).
"""


def roles():
    r = {}
    for n, ids in F.FORWARD.items():
        for i in ids:
            r.setdefault(i, []).append(f"forward ramps {n}/um (Figs. 2-4, 6; exponents)")
    for n, ids in F.REVERSE.items():
        for i in ids:
            r.setdefault(i, []).append(f"reverse ramps {n}/um (Figs. 3, 4; melting exponent)")
    for i, role in V2_ROLES.items():
        r.setdefault(i, []).append(role)
    return r


def cited_runs():
    text = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    ids = set(re.findall(r"2026100[5-8]-\d{6}", text))
    ids |= {i for v in F.FORWARD.values() for i in v} | {i for v in F.REVERSE.values() for i in v}
    out = []
    for d in sorted((ROOT / "runs").glob("2026100[5-8]-*_tube_ramp_3d")):
        rid = d.name.split("_")[0]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        prm = meta.get("params", {})
        used = (rid in ids
                or (prm.get("a_c") == 0 and prm.get("density") == 6250)              # every hold (Fig. 5, rates)
                or (prm.get("noise") == "thermal" and prm.get("density") == 6250))   # thermal ramps (ramp test)
        if meta.get("status") == "finished" and used:
            out.append((rid, d, meta))
    return out


def protocol(p):
    if p.get("a_c") == 0:
        return f"hold at {p['a_f']} a0 after a {p['tau_ms']} ms ramp from {p['a_i']} a0"
    direction = "reverse (crystal start)" if p.get("initial") == "crystal" else "forward"
    return f"{direction} ramp {p['a_i']} -> {p['a_f']} a0"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    rl = roles()
    rows = []
    for rid, d, meta in cited_runs():
        dst = OUT / rid
        dst.mkdir()
        keep = {k: meta.get(k) for k in ("id", "name", "params", "command", "git_commit", "git_dirty",
                                        "environment", "start", "end", "wall_s")}
        (dst / "params.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
        p = meta["params"]
        for f in sorted((d / "data").glob("tau_*ms.pt")):
            D = torch.load(f)
            arrays = {}
            for k, dt in FIELDS.items():
                if k in D:
                    v = D[k]
                    arrays[k] = (v.numpy() if hasattr(v, "numpy") else np.asarray(v)).astype(dt)
            arrays["dz_um"] = np.float64(D["dz_um"])
            np.savez_compressed(dst / (f.stem + ".npz"), **arrays)
            rows.append({"run": rid, "file": f"{rid}/{f.stem}.npz", "density_per_um": p["density"],
                         "tau_ms": float(f.stem[4:-2]), "protocol": protocol(p), "noise": p["noise"],
                         "T_nK": p.get("T_nK") if p["noise"] == "thermal" else 0,
                         "noise_cutoff": p.get("noise_cutoff", 1.0), "dt_safety": p.get("dt_safety"),
                         "realizations": int(D["fs"].shape[1]), "seed": p["seed"],
                         "role": "; ".join(rl.get(rid, ["cited in the paper / docs RESULTS"]))})
        print(rid, protocol(p), flush=True)
    st = OUT / "static"
    st.mkdir()
    for rid in STATIC:
        d = next((ROOT / "runs").glob(rid + "_*"))
        dst = st / d.name
        dst.mkdir()
        for f in d.iterdir():
            if f.suffix in (".json", ".npz"):
                shutil.copy(f, dst / f.name)
        for f in (d / "data").glob("*.pt") if (d / "data").exists() else []:
            S = torch.load(f)
            np.savez_compressed(dst / (f.stem + ".npz"), **{
                f"{k}__{kk}": (vv.numpy() if hasattr(vv, "numpy") else np.asarray(vv))
                for k, v in S.items() for kk, vv in v.items()})
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    lines = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(OUT).as_posix()}")
    (OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    z = ROOT / "data_C1.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                zf.write(f, Path("data_C1") / f.relative_to(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(rows)} files from {len({r['run'] for r in rows})} runs; folder {size / 1e6:.1f} MB; "
          f"{z.name} {z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

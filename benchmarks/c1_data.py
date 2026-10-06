"""Load the recorded observables of a tube-ramp run, either from the original run
record (runs/<id>/data/tau_*ms.pt) or, when those fields are absent (public
release), from the reduced archive data_C1/<id>/tau_*ms.npz written by
tools/export_c1_data.py. Both return {tau_ms: dict} with the same keys
(times_ms, fs, contrast, peaks, deep_peaks, fs_avg_1ms, g2, g2_times_ms, line,
dz_um, args). C1_ARCHIVE_ONLY=1 forces the archive (to test the public path)."""

import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data_C1"
LISTS = ("times_ms", "g2_times_ms")


def _from_npz(path, params):
    z = np.load(path)
    D = {}
    for k in z.files:
        v = z[k]
        if k in LISTS:
            D[k] = v.astype(float).tolist()
        elif k == "dz_um":
            D[k] = float(v)
        else:
            D[k] = torch.from_numpy(v.astype(np.int64) if v.dtype == np.int16 else v)
    args = dict(params)
    args["tau"] = float(path.stem[4:-2])
    D["args"] = args
    return D


def load_tau_files(run_dir):
    """{tau: observables} of one tube_ramp_3d run directory."""
    run_dir = Path(run_dir)
    pts = sorted((run_dir / "data").glob("tau_*ms.pt")) if (run_dir / "data").exists() else []
    if os.environ.get("C1_ARCHIVE_ONLY"):          # test the public-release path
        pts = []
    if pts:
        return {float(f.stem[4:-2]): torch.load(f) for f in pts}
    rid = run_dir.name.split("_")[0]
    arch = ARCHIVE / rid
    if not arch.exists():
        raise FileNotFoundError(f"no fields for {run_dir.name}: neither {run_dir / 'data'} nor {arch}")
    params = json.loads((arch / "params.json").read_text(encoding="utf-8"))["params"]
    return {float(f.stem[4:-2]): _from_npz(f, params) for f in sorted(arch.glob("tau_*ms.npz"))}


def load_states(run_dir):
    """Ground states of tube_states.py: from data/states.pt or the archive."""
    run_dir = Path(run_dir)
    pt = run_dir / "data" / "states.pt"
    if pt.exists() and not os.environ.get("C1_ARCHIVE_ONLY"):
        return torch.load(pt)
    npz = ARCHIVE / "static" / run_dir.name / "states.npz"
    z = np.load(npz)
    out = {}
    for key in z.files:
        name, field = key.split("__", 1)
        v = z[key]
        out.setdefault(name, {})[field] = float(v) if v.ndim == 0 else torch.from_numpy(v)
    return out

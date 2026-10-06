"""Snapshots with enough metadata to reuse them without rerunning."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .grid import Grid


def _git_state():
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def save_snapshot(path, grid, psi, t, params=None):
    """Write psi and metadata (grid, time, params, versions, git state) to an .npz file."""
    meta = {
        "shape": list(grid.shape),
        "length": list(grid.length),
        "dtype": str(grid.dtype).removeprefix("torch."),
        "t": float(t),
        "params": params or {},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__},
        "git": _git_state(),
        "saved": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, psi=psi.detach().cpu().numpy(), meta=json.dumps(meta))
    return path


def load_snapshot(path, device="cpu"):
    """Return (grid, psi, meta) from a file written by :func:`save_snapshot`."""
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        psi = torch.from_numpy(data["psi"])
    grid = Grid(meta["shape"], meta["length"], dtype=getattr(torch, meta["dtype"]), device=device)
    return grid, psi.to(device), meta

"""Live progress, stored frames and a browser viewer for dipgpe runs.

A :class:`LiveRun` writes one run directory:
  meta.json            title, kind, parameters, progress label, git state
  status.json          latest progress, current step, latest metrics
  history.jsonl        one line per checkpoint (progress value, wall time, metrics)
  frames/index.jsonl   one line per stored frame (header: time, shape, scales, files)
  frames/f*_*.bin      the frame data (see :func:`encode_frame`)
  frames/latest.*      the most recent field reported by :meth:`LiveRun.report`

Progress has two levels: :meth:`LiveRun.checkpoint` advances the main progress
axis (simulation time, or tasks done in a benchmark), adds a history row and
stores a frame; :meth:`LiveRun.stage` and :meth:`LiveRun.report` describe the
current step (e.g. a ground state converging inside one task) without touching
the history. ``dipgpe.ground_state(..., live=run)`` reports its convergence this way.

View with ``python -m dipgpe.live serve results/`` (all runs: dashboard, run view,
compare view) or ``LiveRun(..., port=8765)``, which serves the run's parent
directory from a daemon thread. ``python -m dipgpe.live export RUN`` writes a
self-contained HTML, a GIF/MP4 or a PNG. Live output never stops a simulation:
a failed write is skipped with one warning.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
FRAME_SIZE = 256        # max points per axis of stored 2D fields / slices
VOLUME_SIZE = 48        # max points per axis of stored 3D density volumes
MAX_POINTS = 20000      # max vortex points stored per frame


# --------------------------------------------------------------------------- files

def _replace(tmp, path, attempts=20):
    """os.replace, retried: on Windows it fails while the viewer's server has the
    destination open."""
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01)


def _write_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")   # Windows' default cp1252 garbles the viewer
    else:
        tmp.write_bytes(data)
    _replace(tmp, path)


def _git_state():
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                              text=True, check=True).stdout.strip()
        return head or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- frames

def _pool2(x, size):
    """Average-pool the trailing 2 axes of a real tensor down to <= size per axis;
    returns (pooled, stride)."""
    f = max(1, math.ceil(max(x.shape[-2:]) / size))
    if f == 1:
        return x, 1
    return torch.nn.functional.avg_pool2d(x[None, None], f, f, ceil_mode=True)[0, 0], f


def _encode_plane(plane, frame_size):
    """(amplitude, phase) of a 2D complex slice, pooled / strided to <= frame_size."""
    rho, f = _pool2(plane.real ** 2 + plane.imag ** 2, frame_size)
    amp = torch.sqrt(rho).double().cpu().numpy()
    phase = torch.angle(plane[::f, ::f])[: amp.shape[0], : amp.shape[1]].double().cpu().numpy()
    return amp, phase


def _quantize(amp, phase):
    amp_max = float(amp.max()) or 1.0
    return (amp_max, np.round(amp / amp_max * 65535).astype(np.uint16).tobytes(),
            np.round((phase + math.pi) / (2 * math.pi) * 255).clip(0, 255).astype(np.uint8).tobytes())


def encode_frame(grid, psi, vortices=None, frame_size=FRAME_SIZE):
    """Encode one field for the viewer. Returns (header, {suffix: bytes}).

    amplitude |psi| (from the pooled density) as uint16 scaled by ``amp_max``;
    phase as uint8 over [-pi, pi) (strided subsample); for 3D fields the three
    mid-planes (xy at z = mid as the main field; "xz" and "yz" in header["planes"]
    with files amp_xz / phase_xz / amp_yz / phase_yz), a <= VOLUME_SIZE^3 density
    volume (uint8 scaled by ``volume_max``) and vortex points (float32 x, y, z); for
    2D fields vortex positions (float32 x, y) and charges (int8). Arrays are
    row-major with the first listed axis as rows.
    """
    field = psi.detach()
    header = {"kind": f"{grid.ndim}d", "box": list(grid.length)}
    files = {}
    if grid.ndim == 1:
        f = max(1, math.ceil(field.shape[0] / frame_size))
        rho = (field.real ** 2 + field.imag ** 2)
        rho = rho[: (rho.shape[0] // f) * f].reshape(-1, f).mean(1) if f > 1 else rho
        amp = torch.sqrt(rho).double().cpu().numpy()
        phase = torch.angle(field[::f][: amp.shape[0]]).double().cpu().numpy()
        header.update(shape=[amp.shape[0]], extent=[-grid.length[0] / 2, grid.length[0] / 2])
    else:
        plane = field[:, :, field.shape[2] // 2] if grid.ndim == 3 else field
        amp, phase = _encode_plane(plane, frame_size)
        Lx, Ly = grid.length[0], grid.length[1]
        header.update(shape=list(amp.shape), extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2])
        if grid.ndim == 3:
            mid = [float(grid.x64[i].reshape(-1)[field.shape[i] // 2]) for i in range(3)]
            header["slice_z"] = mid[2]
            L = grid.length
            planes = {"xy": {"axes": ["x", "y"], "at": ["z", mid[2]], "shape": list(amp.shape),
                             "extent": [-L[0] / 2, L[0] / 2, -L[1] / 2, L[1] / 2]}}
            for key, slab, axes, at, (a, b) in (
                    ("xz", field[:, field.shape[1] // 2, :], ["x", "z"], ["y", mid[1]], (0, 2)),
                    ("yz", field[field.shape[0] // 2, :, :], ["y", "z"], ["x", mid[0]], (1, 2))):
                pa, pp = _encode_plane(slab, frame_size)
                pmax, files[f"amp_{key}"], files[f"phase_{key}"] = _quantize(pa, pp)
                planes[key] = {"axes": axes, "at": at, "shape": list(pa.shape), "amp_max": pmax,
                               "extent": [-L[a] / 2, L[a] / 2, -L[b] / 2, L[b] / 2]}
            header["planes"] = planes
            dens = field.real ** 2 + field.imag ** 2
            fv = max(1, math.ceil(max(dens.shape) / VOLUME_SIZE))
            vol = torch.nn.functional.avg_pool3d(dens[None, None], fv, fv, ceil_mode=True)[0, 0] if fv > 1 else dens
            vmax = float(vol.max()) or 1.0
            files["volume"] = np.round(vol.double().cpu().numpy() / vmax * 255).astype(np.uint8).tobytes()
            header.update(volume_shape=list(vol.shape), volume_max=vmax)
    amp_max, files["amp"], files["phase"] = _quantize(amp, phase)
    header["amp_max"] = amp_max
    if "planes" in header:
        header["planes"]["xy"]["amp_max"] = amp_max
    if vortices is not None and len(vortices[0]):
        points = vortices[0].detach().cpu().numpy().astype(np.float32)
        if len(points) > MAX_POINTS:
            points = points[np.linspace(0, len(points) - 1, MAX_POINTS).astype(int)]
        files["points"] = points.tobytes()
        header["n_points"] = int(len(points))
        header["point_dim"] = int(points.shape[1])
        if grid.ndim == 2 and len(vortices) > 1:
            charges = vortices[1].detach().cpu().numpy()
            if len(charges) > MAX_POINTS:
                charges = charges[np.linspace(0, len(charges) - 1, MAX_POINTS).astype(int)]
            files["charges"] = charges.astype(np.int8).tobytes()
    return header, files


def decode_frame(frame_dir, header):
    """Inverse of :func:`encode_frame` for a stored header (numpy arrays)."""
    frame_dir = Path(frame_dir)
    shape = tuple(header["shape"])
    read = lambda key, dtype: np.frombuffer((frame_dir / header["files"][key]).read_bytes(), dtype=dtype)
    out = {"amp": read("amp", np.uint16).reshape(shape) / 65535.0 * header["amp_max"],
           "phase": read("phase", np.uint8).reshape(shape) / 255.0 * 2 * math.pi - math.pi}
    for key, plane in header.get("planes", {}).items():
        if key != "xy":
            sh = tuple(plane["shape"])
            out[f"amp_{key}"] = read(f"amp_{key}", np.uint16).reshape(sh) / 65535.0 * plane["amp_max"]
            out[f"phase_{key}"] = read(f"phase_{key}", np.uint8).reshape(sh) / 255.0 * 2 * math.pi - math.pi
    if "volume" in header["files"]:
        out["density_volume"] = (read("volume", np.uint8).reshape(tuple(header["volume_shape"]))
                                 / 255.0 * header["volume_max"])
    if "points" in header["files"]:
        out["points"] = read("points", np.float32).reshape(-1, header["point_dim"])
    if "charges" in header["files"]:
        out["charges"] = read("charges", np.int8)
    return out


# --------------------------------------------------------------------------- run

class LiveRun:
    """Progress, history and frames of one run (see module docstring).

    ``total_time`` enables the fraction-done and ETA; ``progress_label`` names the
    progress axis ("t", "task", ...); ``params`` (e.g. ``vars(args)``) is shown in
    the dashboard and compare tables. ``port`` serves the run's parent directory.
    """

    def __init__(self, directory, title="dipgpe run", total_time=None, port=None, progress_label="t",
                 params=None, frame_size=FRAME_SIZE, preview_size=None):
        self.dir = Path(directory)
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.frames_dir.glob("*"):
            stale.unlink(missing_ok=True)
        self.title, self.total_time, self.progress_label = title, total_time, progress_label
        self.frame_size = preview_size or frame_size
        self.start = time.perf_counter()
        self._progress, self._stage, self._current = {"state": "starting", "t": 0.0}, "", {}
        self._warned, self._frames, self._last_latest = False, 0, 0.0
        clean = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in (params or {}).items()}
        meta = {"title": title, "progress_label": progress_label, "total": total_time, "params": clean,
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "git": _git_state(), "format": 2}
        self._safely(_write_atomic, self.dir / "meta.json", json.dumps(meta))
        self._history = open(self.dir / "history.jsonl", "w", encoding="utf-8")
        self._index = open(self.frames_dir / "index.jsonl", "w", encoding="utf-8")
        self.server = serve(self.dir.parent, port, background=True) if port is not None else None
        self._status()

    @property
    def url(self):
        if self.server is None:
            return None
        return f"http://localhost:{self.server.server_address[1]}/#run={self.dir.name}"

    def _safely(self, fn, *args):
        """Live output must never stop the simulation: report the first failure, skip it."""
        try:
            fn(*args)
        except OSError as error:
            if not self._warned:
                print(f"dipgpe.live: skipped an update ({error}); the run continues")
                self._warned = True

    def _status(self, **extra):
        self._progress.update(extra)
        status = {"title": self.title, "total_time": self.total_time, "progress_label": self.progress_label,
                  "wall_s": time.perf_counter() - self.start, "stage": self._stage,
                  "current": self._current, "frames": self._frames, **self._progress}
        self._safely(_write_atomic, self.dir / "status.json", json.dumps(status))

    def _write_frame(self, name, header, files):
        header = dict(header)
        header["files"] = {}
        for key, data in files.items():
            fname = f"{name}_{key}.bin"
            self._safely(_write_atomic, self.frames_dir / fname, data)
            header["files"][key] = fname
        return header

    def stage(self, text):
        """Describe what the run is doing now."""
        self._stage, self._current = str(text), {}
        self._status()

    def report(self, metrics=None, grid=None, psi=None, vortices=None):
        """Update the current step's metrics and the 'latest' field (at most once a
        second) without adding history or a stored frame."""
        self._current = {k: float(v) for k, v in (metrics or {}).items()}
        now = time.perf_counter()
        if grid is not None and psi is not None and now - self._last_latest > 1.0:
            self._last_latest = now
            header, files = encode_frame(grid, psi, vortices, self.frame_size)
            header = self._write_frame("latest", header, files)
            header.update(t=float(self._progress.get("t", 0.0)), stage=self._stage)
            self._safely(_write_atomic, self.frames_dir / "latest.json", json.dumps(header))
        self._status()

    def checkpoint(self, t, metrics=None, grid=None, psi=None, vortices=None):
        """Record progress value t and metrics; with grid and psi also store a frame.

        ``vortices`` (optional) is (positions, charges) from find_vortices for 2D,
        or (points,) from find_vortex_points for 3D.
        """
        wall = time.perf_counter() - self.start
        metrics = {k: float(v) for k, v in (metrics or {}).items()}
        row = {"t": float(t), "wall_s": wall, **metrics}
        self._safely(lambda: (self._history.write(json.dumps(row) + "\n"), self._history.flush()))
        if grid is not None and psi is not None:
            header, files = encode_frame(grid, psi, vortices, self.frame_size)
            header = self._write_frame(f"f{self._frames:05d}", header, files)
            header.update(i=self._frames, t=float(t), stage=self._stage, wall_s=wall)
            self._safely(lambda: (self._index.write(json.dumps(header) + "\n"), self._index.flush()))
            self._frames += 1
        status = {"state": "running", "t": float(t), "metrics": metrics}
        if self.total_time:
            fraction = min(1.0, float(t) / self.total_time)
            status["fraction"] = fraction
            if fraction > 0:
                status["eta_s"] = wall * (1 - fraction) / fraction
        self._status(**status)

    def close(self):
        """Finish the run. The in-process server stops with the process; view the
        finished run with ``python -m dipgpe.live serve DIR``."""
        self._history.close()
        self._index.close()
        self._stage, self._current = "", {}
        final = {"state": "finished", "t": self.total_time if self.total_time else self._progress.get("t", 0.0)}
        if self.total_time:
            final["fraction"] = 1.0
        self._status(**final)
        if self.server is not None:
            print(f"live view stops with this process; to keep viewing: uv run python -m dipgpe.live serve {self.dir.parent}")


# --------------------------------------------------------------------------- server

def find_runs(root):
    """Run directories (containing status.json) at depth <= 2 below root."""
    root = Path(root)
    found = []
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    for c in list(candidates):
        if c != root:
            candidates += [p for p in c.iterdir() if p.is_dir() and p.name != "frames"]
    for c in candidates:
        if (c / "status.json").exists():
            found.append(c)
    return found


def run_summary(root, run_dir):
    def load(name):
        try:
            return json.loads((run_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    name = "." if run_dir == Path(root) else run_dir.relative_to(root).as_posix()
    return {"name": name, "meta": load("meta.json") or {}, "status": load("status.json") or {},
            "modified": (run_dir / "status.json").stat().st_mtime}


class _Handler(SimpleHTTPRequestHandler):
    """Static files from the served root, the viewer at / and /_viewer/, and /api/runs."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/runs":
            root = Path(self.directory)
            body = json.dumps([run_summary(root, r) for r in find_runs(root)]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path):
        p = unquote(urlparse(path).path)
        if p in ("/", "/index.html"):
            return str(VIEWER_DIR / "index.html")
        if p.startswith("/_viewer/"):
            return str(VIEWER_DIR / p[len("/_viewer/"):])
        return super().translate_path(path)


def serve(directory, port=8765, background=False):
    """Serve ``directory`` (a run or a folder of runs) on localhost; port 0 picks a free port."""
    handler = functools.partial(_Handler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    else:
        print(f"serving {directory} at http://localhost:{server.server_address[1]} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return server


# --------------------------------------------------------------------------- cli

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("serve", "export", "-h", "--help"):
        argv = ["serve"] + argv                      # backwards compatible: python -m dipgpe.live DIR
    ap = argparse.ArgumentParser(prog="python -m dipgpe.live", description="View or export dipgpe runs.")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="serve a run or a folder of runs (default: results)")
    s.add_argument("directory", type=Path, nargs="?", default=Path("results"))
    s.add_argument("--port", type=int, default=8765)
    e = sub.add_parser("export", help="export a run as HTML, GIF/MP4 and/or PNG")
    e.add_argument("run", type=Path)
    e.add_argument("--html", type=Path, help="self-contained HTML file")
    e.add_argument("--max-frames", type=int, default=60, help="frames embedded in the HTML (evenly spaced)")
    e.add_argument("--video", type=Path, help=".gif or .mp4 (MP4 needs ffmpeg)")
    e.add_argument("--png", type=Path, help="summary picture: last frame and metric curves")
    e.add_argument("--field", choices=["density", "phase"], default="density")
    e.add_argument("--fps", type=int, default=8)
    e.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy", help="mid-plane of 3D runs")
    args = ap.parse_args(argv)
    if args.command == "serve":
        serve(args.directory, args.port)
        return
    from .live_export import export_html, export_png, export_video
    if not (args.html or args.video or args.png):
        args.html = args.run.with_suffix(".html")
    if args.html:
        path = export_html(args.run, args.html, args.max_frames)
        size = path.stat().st_size / 2 ** 20
        note = " (over 15 MB: lower --max-frames to publish it as a claude.ai artifact)" if size > 15 else ""
        print(f"wrote {path} ({size:.1f} MB){note}")
    if args.video:
        print("wrote", export_video(args.run, args.video, args.field, args.fps, args.plane))
    if args.png:
        print("wrote", export_png(args.run, args.png, args.field, args.plane))


if __name__ == "__main__":
    main()

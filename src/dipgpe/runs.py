"""Permanent records of simulation runs.

Every simulation runs inside ``with dipgpe.runs.Run(name, params) as run:``. The run
gets its own directory ``runs/<YYYYMMDD-HHMMSS>_<name>`` that is never reused or
overwritten, holding

  meta.json     command, parameters, git commit (and ``git.diff`` if the tree
                was dirty), Python / torch / CUDA / GPU / driver versions,
                host, start and end time, wall time, status (running /
                finished / failed, with the traceback)
  result.json   the run's key numbers (``run.result(...)``)
  log.txt       everything printed to stdout and stderr during the run
  *.json, *.png small outputs (``run.file(name)``): committed to git
  data/         large outputs: fields, time series, checkpoints
                (``run.data(name)``): git-ignored, copied by ``backup``
  MANIFEST      SHA256 and size of every file, written when the run ends

and one line in ``runs/index.jsonl`` (name, status, commit, parameters,
result) so that earlier runs can be found instead of repeated:

  uv run python -m dipgpe.runs list [name]
  uv run python -m dipgpe.runs find NAME key=value ...
  uv run python -m dipgpe.runs show RUN_ID
  uv run python -m dipgpe.runs backup DEST        (or QS_RUNS_BACKUP)
  uv run python -m dipgpe.runs repair             (mark killed runs as interrupted)
  uv run python -m dipgpe.runs note RUN_ID TEXT [--invalid]

``Run`` prints the ids of finished runs with the same name and parameters, and
``find`` / ``latest`` return them, so a script can reuse a stored result.
The runs directory is ``QS_RUNS`` if set, else ``runs/`` in the repository.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def runs_dir():
    return Path(os.environ.get("QS_RUNS", REPO / "runs"))


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, encoding="utf-8", errors="replace",
                             timeout=30)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _environment():
    env = {"python": platform.python_version(), "platform": platform.platform(), "host": socket.gethostname()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name()
            smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, encoding="utf-8", errors="replace", timeout=30)
            env["driver"] = smi.stdout.strip() or None
    except Exception:                                    # environment info must never stop a run
        pass
    return env


def _jsonable(value):
    """Parameters as JSON: paths and other objects become strings."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Run:
    """Context manager recording one simulation run (see the module docstring)."""

    def __init__(self, name, params=None, *, root=None, note=None, quiet=False):
        self.name = name.replace(" ", "_")
        self.params = _jsonable(params or {})
        self.root = Path(root) if root is not None else runs_dir()
        self.note = note
        self.quiet = quiet
        self.dir = None
        self._result = {}

    # ----------------------------------------------------------------- paths
    def file(self, name):
        """Path for a small output kept in git (JSON, figures)."""
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def data(self, name):
        """Path for a large output under data/ (git-ignored, backed up)."""
        p = self.dir / "data" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def result(self, **values):
        """Merge key numbers into result.json (written now and at the end)."""
        self._result.update(_jsonable(values))
        (self.dir / "result.json").write_text(json.dumps(self._result, indent=1), encoding="utf-8", newline="\n")

    def save_json(self, name, obj, also=None, large=False, indent=1):
        """Write obj as JSON into the run (data/ if large) and, if ``also`` is a
        path, a copy there; returns the path inside the run."""
        path = self.data(name) if large else self.file(name)
        text = json.dumps(obj, indent=indent)
        path.write_text(text, encoding="utf-8", newline="\n")
        if also is not None:
            Path(also).parent.mkdir(parents=True, exist_ok=True)
            Path(also).write_text(text, encoding="utf-8")
        return path

    def live(self, title, total_time=None, port=None, **kw):
        """A dipgpe.live.LiveRun inside this run (directory live/)."""
        from .live import LiveRun
        kw.setdefault("params", self.params)
        return LiveRun(self.dir / "live", title, total_time, port, **kw)

    # -------------------------------------------------------------- lifecycle
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{self.name}"
        self.dir = self.root / base
        k = 1
        while self.dir.exists():
            k += 1
            self.dir = self.root / f"{base}_{k}"
        self.dir.mkdir(parents=True)
        self.id = self.dir.name
        commit = (_git("rev-parse", "HEAD") or "").strip() or None
        status = _git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status and status.strip())
        if dirty:
            (self.dir / "git.diff").write_text(_git("diff", "HEAD") or "", encoding="utf-8", newline="\n")
        self.meta = {"id": self.id, "name": self.name, "params": self.params, "note": self.note,
                     "command": [sys.executable] + sys.argv, "cwd": os.getcwd(),
                     "git_commit": commit, "git_dirty": dirty, "environment": _environment(),
                     "start": _dt.datetime.now().isoformat(timespec="seconds"), "status": "running"}
        self._write_meta()
        self._log = open(self.dir / "log.txt", "w", encoding="utf-8", newline="\n")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._log), _Tee(sys.stderr, self._log)
        self._t0 = time.perf_counter()
        same = [r["id"] for r in find(self.name, root=self.root, **self.params) if r["id"] != self.id]
        if same and not self.quiet:
            print(f"[dipgpe.runs] finished runs with the same name and parameters: {', '.join(same[-3:])}")
        if not self.quiet:
            print(f"[dipgpe.runs] recording to {self.dir}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.meta["wall_s"] = time.perf_counter() - self._t0
        self.meta["end"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.meta["status"] = "failed" if exc_type else "finished"
        if exc_type:
            self.meta["error"] = "".join(traceback.format_exception(exc_type, exc, tb))
            print(self.meta["error"], file=sys.stderr)
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._log.close()
        if self._result:
            (self.dir / "result.json").write_text(json.dumps(self._result, indent=1), encoding="utf-8", newline="\n")
        self._write_meta()
        self._write_manifest()
        entry = {k: self.meta[k] for k in ("id", "name", "status", "start", "end", "wall_s", "git_commit",
                                           "git_dirty", "params")}
        entry["result"] = self._result
        with open(self.root / "index.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry) + "\n")
        return False

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8", newline="\n")

    def _write_manifest(self):
        lines = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.name != "MANIFEST":
                lines.append(f"{_sha256(p)}  {p.stat().st_size:>12d}  {p.relative_to(self.dir).as_posix()}")
        (self.dir / "MANIFEST").write_text("\n".join(lines) + "\n", encoding="utf-8")


def register(name, files, *, params=None, note=None, root=None, large_bytes=1 << 20, when=None, commit=None):
    """Record existing output files as a finished run (e.g. results from before
    dipgpe.runs). Files and directories are copied; files above ``large_bytes`` and
    directories go to data/. ``when`` (datetime) and ``commit`` describe when and
    from which code the files were produced, if known."""
    root = Path(root) if root is not None else runs_dir()
    when = when or _dt.datetime.now()
    base = f"{when.strftime('%Y%m%d-%H%M%S')}_{name}"
    d = root / base
    k = 1
    while d.exists():
        k += 1
        d = root / f"{base}_{k}"
    d.mkdir(parents=True)
    for f in map(Path, files):
        if f.is_dir():
            shutil.copytree(f, d / "data" / f.name)
        elif f.stat().st_size > large_bytes:
            (d / "data").mkdir(exist_ok=True)
            shutil.copy2(f, d / "data" / f.name)
        else:
            shutil.copy2(f, d / f.name)
    meta = {"id": d.name, "name": name, "params": _jsonable(params or {}), "note": note,
            "command": None, "git_commit": commit, "git_dirty": None, "environment": None,
            "start": when.isoformat(timespec="seconds"), "end": when.isoformat(timespec="seconds"),
            "wall_s": None, "status": "finished", "registered": _dt.datetime.now().isoformat(timespec="seconds"),
            "sources": [str(f) for f in files]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    run = Run.__new__(Run)
    run.dir = d
    run._write_manifest()
    entry = {k_: meta[k_] for k_ in ("id", "name", "status", "start", "end", "wall_s", "git_commit", "git_dirty",
                                     "params")}
    entry["result"] = {}
    entry["registered"] = True
    with open(root / "index.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    return d


# ------------------------------------------------------------------- queries
def _lines(root=None):
    path = Path(root or runs_dir()) / "index.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(root=None):
    """Run entries of the index, each with the notes added later and an
    ``invalid`` flag (see :func:`annotate`)."""
    rows, notes = [], {}
    for r in _lines(root):
        if r.get("event") == "note":
            notes.setdefault(r["id"], []).append(r)
        else:
            rows.append(r)
    for r in rows:
        r["notes"] = [n["note"] for n in notes.get(r["id"], [])]
        r["invalid"] = any(n.get("invalid") for n in notes.get(r["id"], []))
    return rows


def annotate(run_id, note, invalid=False, root=None):
    """Attach a note to a run (meta.json and an index event); ``invalid=True``
    marks its results as wrong (e.g. produced by a bug), so :func:`find` skips it.
    Runs are never edited otherwise or deleted."""
    root = Path(root or runs_dir())
    meta_path = root / run_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entry = {"event": "note", "id": run_id, "note": note, "invalid": bool(invalid),
             "time": _dt.datetime.now().isoformat(timespec="seconds")}
    meta.setdefault("notes", []).append({k: entry[k] for k in ("note", "invalid", "time")})
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    with open(root / "index.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")


def _matches(value, wanted):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return abs(float(value) - float(wanted)) <= 1e-12 * max(1.0, abs(float(value)))
        except (TypeError, ValueError):
            return False
    return value == wanted or str(value) == str(wanted)


def find(name=None, root=None, status="finished", **params):
    """Index entries with this name (if given), status and parameter values."""
    out = []
    for r in index(root):
        if name is not None and r["name"] != name:
            continue
        if status is not None and r["status"] != status:
            continue
        if r["invalid"] and status is not None:
            continue
        if all(k in r["params"] and _matches(r["params"][k], v) for k, v in params.items()):
            out.append(r)
    return out


def latest(name, root=None, **params):
    """Directory of the most recent finished run matching ``find`` (or None)."""
    rows = find(name, root=root, **params)
    return Path(root or runs_dir()) / rows[-1]["id"] if rows else None


def resolve(ref, filename=None, root=None):
    """A run id (or directory) and optional file name -> path; plain paths pass through."""
    p = Path(ref)
    if not p.exists():
        p = Path(root or runs_dir()) / ref
    if filename is not None and p.is_dir():
        for cand in (p / filename, p / "data" / filename):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"{filename} not in run {ref}")
    return p


def backup(dest, root=None):
    """Copy every run (including data/) to dest, skipping files that are already
    there with the same size and SHA256; returns (copied, skipped)."""
    src, dest = Path(root or runs_dir()), Path(dest)
    copied = skipped = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        target = dest / p.relative_to(src)
        if target.exists() and target.stat().st_size == p.stat().st_size and _sha256(target) == _sha256(p):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied += 1
    return copied, skipped


def repair(root=None):
    """Runs whose process was killed (meta status 'running', not in the index) are
    marked 'interrupted', given a MANIFEST and appended to the index; returns ids."""
    root = Path(root or runs_dir())
    listed = {r["id"] for r in index(root)}
    fixed = []
    for meta_path in sorted(root.glob("*/meta.json")):
        d = meta_path.parent
        if d.name in listed:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "running":
            continue
        meta["status"] = "interrupted"
        meta["end"] = _dt.datetime.fromtimestamp(max(p_.stat().st_mtime for p_ in d.rglob("*") if p_.is_file())
                                                 ).isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
        run = Run.__new__(Run)
        run.dir = d
        run._write_manifest()
        entry = {k: meta.get(k) for k in ("id", "name", "status", "start", "end", "wall_s", "git_commit",
                                          "git_dirty", "params")}
        entry["result"] = {}
        with open(root / "index.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry) + "\n")
        fixed.append(d.name)
    return fixed


def _parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dipgpe.runs", description="Find and back up recorded runs.")
    ap.add_argument("--root", default=None, help="runs directory (default: QS_RUNS or ./runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("name", nargs="?")
    s = sub.add_parser("find"); s.add_argument("name"); s.add_argument("params", nargs="*", help="key=value")
    s = sub.add_parser("show"); s.add_argument("run")
    s = sub.add_parser("backup"); s.add_argument("dest", nargs="?", default=os.environ.get("QS_RUNS_BACKUP"))
    sub.add_parser("repair", help="mark runs whose process was killed as interrupted")
    s = sub.add_parser("note", help="attach a note to a run")
    s.add_argument("run"); s.add_argument("text"); s.add_argument("--invalid", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd in ("list", "find"):
        params = dict(kv.split("=", 1) for kv in getattr(args, "params", []) or [])
        rows = find(args.name, root=args.root, status=None, **{k: _parse_value(v) for k, v in params.items()})
        for r in rows:
            res = ", ".join(f"{k}={v}" for k, v in list(r.get("result", {}).items())[:4])
            wall = r.get("wall_s") or 0.0
            flag = "INVALID " if r["invalid"] else ""
            print(f"{r['id']:48s} {r['status']:9s} {wall:8.0f} s  {(r['git_commit'] or '')[:8]}"
                  f"{'+' if r.get('git_dirty') else ' '} {flag}{res}")
    elif args.cmd == "show":
        d = resolve(args.run, root=args.root)
        print((d / "meta.json").read_text(encoding="utf-8"))
        if (d / "result.json").exists():
            print((d / "result.json").read_text(encoding="utf-8"))
    elif args.cmd == "note":
        annotate(args.run, args.text, invalid=args.invalid, root=args.root)
    elif args.cmd == "repair":
        for rid in repair(args.root):
            print("interrupted:", rid)
    elif args.cmd == "backup":
        if not args.dest:
            ap.error("give DEST or set QS_RUNS_BACKUP")
        copied, skipped = backup(args.dest, root=args.root)
        print(f"copied {copied} files, {skipped} already up to date -> {args.dest}")


if __name__ == "__main__":
    main()

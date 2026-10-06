"""Export a dipgpe.live run directory: self-contained HTML, GIF/MP4 animation, PNG summary."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

from .live import VIEWER_DIR, decode_frame, run_summary


def _jsonl(path):
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


def export_html(run_dir, output, max_frames=None):
    """One HTML file with the viewer, run metadata, history and the stored frames
    embedded (base64). Opens without a server; Plotly loads from cdnjs.
    ``max_frames`` keeps evenly spaced frames (always the last one) to bound the
    size; a claude.ai artifact must stay under 16 MB."""
    run_dir, output = Path(run_dir), Path(output)
    name = run_dir.name
    summary = run_summary(run_dir.parent, run_dir)
    embed = {"runs": [summary], "json": {}, "text": {}, "bin": {}, "start": f"run={name}"}
    for f in ("status.json", "meta.json"):
        if (run_dir / f).exists():
            embed["json"][f"{name}/{f}"] = json.loads((run_dir / f).read_text(encoding="utf-8"))
    if (run_dir / "history.jsonl").exists():
        embed["text"][f"{name}/history.jsonl"] = (run_dir / "history.jsonl").read_text(encoding="utf-8")
    headers = _jsonl(run_dir / "frames" / "index.jsonl")
    if max_frames and len(headers) > max_frames:
        keep = sorted(set(np.linspace(0, len(headers) - 1, max_frames).round().astype(int).tolist()))
        headers = [headers[i] for i in keep]
    embed["text"][f"{name}/frames/index.jsonl"] = "".join(json.dumps(h) + "\n" for h in headers)
    for header in headers:
        for fname in header["files"].values():
            embed["bin"][f"{name}/frames/{fname}"] = base64.b64encode((run_dir / "frames" / fname).read_bytes()).decode()
    html = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    css = (VIEWER_DIR / "app.css").read_text(encoding="utf-8")
    js = (VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    data = json.dumps(embed).replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="_viewer/app.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="_viewer/app.js"></script>',
                        f"<script>window.QS_EMBED = {data};</script>\n<script>\n{js}\n</script>")
    html = html.replace("<title>dipgpe runs</title>", f"<title>{summary['meta'].get('title', name)}</title>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def _frames(run_dir):
    run_dir = Path(run_dir)
    headers = _jsonl(run_dir / "frames" / "index.jsonl")
    if not headers:
        raise ValueError(f"{run_dir} has no stored frames")
    return headers, [decode_frame(run_dir / "frames", h) for h in headers]


def _plane(header, plane):
    """(key suffix, extent, [vertical, horizontal] axis names) of a stored 2D slice."""
    planes = header.get("planes", {})
    if plane not in (None, "xy") and plane in planes:
        return f"_{plane}", planes[plane]["extent"], planes[plane]["axes"]
    if plane not in (None, "xy"):
        raise ValueError(f"frame has no {plane} plane (3D runs store xy, xz and yz)")
    return "", header["extent"], ["x", "y"]


def _field_image(header, data, field, plane=None):
    suffix = _plane(header, plane)[0]
    if field == "phase":
        return data["phase" + suffix], dict(cmap="twilight", vmin=-math.pi, vmax=math.pi)
    return data["amp" + suffix] ** 2, dict(cmap="viridis")


def export_video(run_dir, output, field="density", fps=8, plane="xy"):
    """Animate the stored frames (2D field, or a mid-plane of 3D runs: xy, xz, yz).
    .gif uses Pillow; .mp4 needs ffmpeg on PATH."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    run_dir, output = Path(run_dir), Path(output)
    headers, frames = _frames(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    label = meta.get("progress_label", "t")
    writer = "pillow" if output.suffix.lower() == ".gif" else "ffmpeg"
    if not animation.writers.is_available(writer):
        raise RuntimeError("MP4 export needs ffmpeg on PATH; use a .gif output instead")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    h0 = headers[0]
    if h0["kind"] == "1d":
        x = np.linspace(*h0["extent"], h0["shape"][0], endpoint=False)
        line, = ax.plot(x, frames[0]["amp"] ** 2 if field == "density" else frames[0]["phase"])
        ax.set_xlabel("x")
        top = max(float((f["amp"] ** 2).max()) for f in frames) if field == "density" else math.pi
        ax.set_ylim(-math.pi if field == "phase" else 0, top * 1.05 if field == "density" else math.pi)

        def update(i):
            line.set_ydata(frames[i]["amp"] ** 2 if field == "density" else frames[i]["phase"])
            ax.set_title(f"{meta.get('title', run_dir.name)}   {label} = {headers[i]['t']:.3g}")
            return (line,)
    else:
        _, (x0, x1, y0, y1), axes = _plane(h0, plane)
        img, kw = _field_image(h0, frames[0], field, plane)
        if field == "density":
            kw["vmax"] = max(float(_field_image(h, f, field, plane)[0].max()) for h, f in zip(headers, frames))
            kw["vmin"] = 0
        im = ax.imshow(img, origin="lower", extent=(y0, y1, x0, x1), **kw)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel(axes[1])
        ax.set_ylabel(axes[0])

        def update(i):
            im.set_data(_field_image(headers[i], frames[i], field, plane)[0])
            ax.set_title(f"{meta.get('title', run_dir.name)}   {label} = {headers[i]['t']:.3g}", fontsize=10)
            return (im,)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer=writer, fps=fps)
    plt.close(fig)
    return output


def export_png(run_dir, output, field="density", plane="xy"):
    """Summary picture: the last stored frame and every metric's history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output = Path(run_dir), Path(output)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    history = _jsonl(run_dir / "history.jsonl")
    keys = [k for k in (history[-1] if history else {}) if k not in ("t", "wall_s")]
    headers = _jsonl(run_dir / "frames" / "index.jsonl")
    n_plots = len(keys) + (1 if headers else 0)
    cols = min(3, max(1, n_plots))
    rows = math.ceil(n_plots / cols) if n_plots else 1
    fig = plt.figure(figsize=(5 * cols, 3.8 * rows), constrained_layout=True)
    k = 1
    if headers:
        h = headers[-1]
        data = decode_frame(run_dir / "frames", h)
        ax = fig.add_subplot(rows, cols, k)
        k += 1
        if h["kind"] == "1d":
            ax.plot(np.linspace(*h["extent"], h["shape"][0], endpoint=False),
                    data["amp"] ** 2 if field == "density" else data["phase"])
        else:
            img, kw = _field_image(h, data, field, plane)
            _, (x0, x1, y0, y1), axes = _plane(h, plane)
            im = ax.imshow(img, origin="lower", extent=(y0, y1, x0, x1), **kw)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel(axes[1])
            ax.set_ylabel(axes[0])
        ax.set_title(f"{field} at {meta.get('progress_label', 't')} = {h['t']:.3g}")
    label = meta.get("progress_label", "t")
    for key in keys:
        ax = fig.add_subplot(rows, cols, k)
        k += 1
        pts = [(r["t"], r[key]) for r in history if isinstance(r.get(key), (int, float))]
        if pts:
            ax.plot(*zip(*pts), marker=".")
        ax.set_title(key)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(meta.get("title", run_dir.name))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return output

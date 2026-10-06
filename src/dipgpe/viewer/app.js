/* DipGPE viewer: dashboard, run view and compare view for DipGPE run directories.
 * Data comes from the dipgpe.live server (fetch) or, in an exported HTML file, from
 * window.QS_EMBED. Frames are decoded from the binary format of dipgpe.live.encode_frame. */
"use strict";

const EMBED = window.QS_EMBED || null;
const POLL_MS = 2000;
const state = { timers: [], cache: new Map(), playing: null };

/* ------------------------------------------------------------------ data access */
const bust = (p) => (EMBED ? p : `${p}${p.includes("?") ? "&" : "?"}_=${Date.now()}`);
async function getJSON(path) {
  if (EMBED) { if (!(path in EMBED.json)) throw new Error("missing " + path); return EMBED.json[path]; }
  const r = await fetch(bust(path)); if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json();
}
async function getText(path) {
  if (EMBED) return EMBED.text[path] || "";
  const r = await fetch(bust(path)); return r.ok ? r.text() : "";
}
async function getBin(path) {
  if (EMBED) {
    const s = atob(EMBED.bin[path]); const b = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
    return b.buffer;
  }
  const r = await fetch(path); if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.arrayBuffer();
}
const jsonl = (text) => text.split("\n").filter((l) => l.trim()).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
async function listRuns() { return EMBED ? EMBED.runs : getJSON("api/runs"); }
const runPath = (name, file) => (name === "." ? file : `${name}/${file}`);

async function loadRun(name) {
  const [status, meta, history, index] = await Promise.all([
    getJSON(runPath(name, "status.json")).catch(() => ({})),
    getJSON(runPath(name, "meta.json")).catch(() => ({})),
    getText(runPath(name, "history.jsonl")).then(jsonl),
    getText(runPath(name, "frames/index.jsonl")).then(jsonl),
  ]);
  return { name, status, meta, history, frames: index };
}

async function loadFrame(name, header) {
  const key = `${name}|${header.files.amp}|${header.t}`;
  if (state.cache.has(key)) return state.cache.get(key);
  const files = {};
  await Promise.all(Object.entries(header.files).map(async ([k, f]) => { files[k] = await getBin(runPath(name, "frames/" + f)); }));
  const frame = decodeFrame(header, files);
  state.cache.set(key, frame);
  if (state.cache.size > 80) state.cache.delete(state.cache.keys().next().value);
  return frame;
}

function decodeFrame(h, files) {
  const amp16 = new Uint16Array(files.amp), ph8 = new Uint8Array(files.phase);
  const n = amp16.length, amp = new Float32Array(n), phase = new Float32Array(n);
  for (let i = 0; i < n; i++) { amp[i] = (amp16[i] / 65535) * h.amp_max; phase[i] = (ph8[i] / 255) * 2 * Math.PI - Math.PI; }
  const f = { header: h, shape: h.shape, amp, phase };
  if (h.planes) {
    f.planes = { xy: { ...h.planes.xy, amp, phase, amp_max: h.amp_max } };
    for (const key of ["xz", "yz"]) {
      const pl = h.planes[key]; if (!pl || !files["amp_" + key]) continue;
      const a16 = new Uint16Array(files["amp_" + key]), p8 = new Uint8Array(files["phase_" + key]);
      const pa = new Float32Array(a16.length), pp = new Float32Array(a16.length);
      for (let i = 0; i < a16.length; i++) { pa[i] = (a16[i] / 65535) * pl.amp_max; pp[i] = (p8[i] / 255) * 2 * Math.PI - Math.PI; }
      f.planes[key] = { ...pl, amp: pa, phase: pp };
    }
  }
  if (files.points) f.points = new Float32Array(files.points);
  if (files.charges) f.charges = new Int8Array(files.charges);
  if (files.volume) f.volume = new Uint8Array(files.volume);
  return f;
}

/* ------------------------------------------------------------------ helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function fmt(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  if (typeof x !== "number") return esc(x);
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return x.toExponential(3);
  return Number.isInteger(x) ? String(x) : x.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
function hms(s) {
  if (s === undefined || s === null) return "–";
  s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s % 60}s`;
}
const ago = (t) => hms(Date.now() / 1000 - t) + " ago";
function runState(run) {
  const st = run.status || {};
  if (st.state === "finished") return "finished";
  if (run.modified && Date.now() / 1000 - run.modified > 60) return "stale";   // no update for a minute
  return st.state || "unknown";
}
function kv(obj) {
  const rows = Object.entries(obj || {}).filter(([, v]) => v !== null && v !== undefined);
  return rows.length ? `<table class="kv">${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fmt(v)}</td></tr>`).join("")}</table>`
    : `<p class="muted">–</p>`;
}
function clearTimers() { state.timers.forEach(clearInterval); state.timers = []; if (state.playing) { clearInterval(state.playing); state.playing = null; } }

/* ------------------------------------------------------------------ theme */
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function plotLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: cssVar("--text"), size: 12, family: "system-ui, sans-serif" },
    margin: { l: 56, r: 16, t: 28, b: 44 },
    xaxis: { gridcolor: cssVar("--border"), zerolinecolor: cssVar("--border") },
    yaxis: { gridcolor: cssVar("--border"), zerolinecolor: cssVar("--border") },
    hovermode: "x unified", legend: { orientation: "h", y: -0.2 },
  }, extra);
}
const PLOT_CONFIG = { responsive: true, displaylogo: false, toImageButtonOptions: { format: "png", scale: 2 } };
const PHASE_SCALE = [[0, "#e2d9e2"], [0.25, "#5e80c2"], [0.5, "#2f1436"], [0.75, "#b45b3e"], [1, "#e2d9e2"]];
function setTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("dipgpe-theme", t || ""); } catch {}
  route();
}
$("#theme").onclick = () => {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  setTheme(dark ? "light" : "dark");
};
try { const t = localStorage.getItem("dipgpe-theme"); if (t) document.documentElement.dataset.theme = t; } catch {}

/* ------------------------------------------------------------------ field rendering */
function sliceOf(frame, plane) {
  const h = frame.header;
  if (frame.planes && plane && frame.planes[plane]) return frame.planes[plane];
  return { amp: frame.amp, phase: frame.phase, shape: frame.shape, extent: h.extent, axes: ["x", "y"], amp_max: h.amp_max };
}
function fieldTraces(frame, opts) {
  const h = frame.header, sl = sliceOf(frame, opts.plane);
  const [rows, cols] = sl.shape.length === 2 ? sl.shape : [sl.shape[0], 1];
  if (h.kind === "1d") {
    const n = frame.shape[0], [x0, x1] = h.extent, xs = Array.from({ length: n }, (_, i) => x0 + (i * (x1 - x0)) / n);
    const y = opts.view === "phase" ? Array.from(frame.phase) : Array.from(frame.amp, (a) => a * a);
    return [{ x: xs, y, type: "scatter", mode: "lines", line: { color: cssVar("--accent") }, name: opts.view }];
  }
  const z = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (opts.view === "phase") row[c] = sl.phase[i];
      else { const d = sl.amp[i] * sl.amp[i]; row[c] = opts.log ? Math.log10(d + 1e-12 * sl.amp_max * sl.amp_max) : d; }
    }
    z[r] = row;
  }
  const [x0, x1, y0, y1] = sl.extent, [va, ha] = sl.axes;
  const ys = Array.from({ length: cols }, (_, c) => y0 + ((c + 0.5) * (y1 - y0)) / cols);   // horizontal: second axis
  const xs = Array.from({ length: rows }, (_, r) => x0 + ((r + 0.5) * (x1 - x0)) / rows);   // vertical: first axis
  const traces = [{
    z, x: ys, y: xs, type: "heatmap", zsmooth: "best",
    colorscale: opts.view === "phase" ? PHASE_SCALE : opts.cmap,
    zmin: opts.view === "phase" ? -Math.PI : undefined, zmax: opts.view === "phase" ? Math.PI : undefined,
    colorbar: { title: { text: opts.view === "phase" ? "arg ψ" : opts.log ? "log₁₀|ψ|²" : "|ψ|²", side: "right" }, thickness: 12 },
    hovertemplate: `${ha} %{x:.3f}<br>${va} %{y:.3f}<br>${opts.view === "phase" ? "arg ψ" : "|ψ|²"} %{z:.4g}<extra></extra>`,
  }];
  if (opts.vortices && frame.points && h.kind === "2d" && frame.charges) {
    const pos = { x: [], y: [] }, neg = { x: [], y: [] };
    for (let i = 0; i < frame.charges.length; i++) {
      const target = frame.charges[i] > 0 ? pos : neg;
      target.y.push(frame.points[2 * i]); target.x.push(frame.points[2 * i + 1]);
    }
    const size = frame.charges.length > 1000 ? 3 : frame.charges.length > 200 ? 4 : 6;
    const marker = (c) => ({ size, color: c, line: { width: 0.5, color: "#000" } });
    traces.push({ ...pos, type: "scattergl", mode: "markers", name: "+1", marker: marker("#ff4d4d"), hoverinfo: "skip" });
    traces.push({ ...neg, type: "scattergl", mode: "markers", name: "−1", marker: marker("#ffffff"), hoverinfo: "skip" });
  }
  return traces;
}
function fieldLayout(frame, title, plane) {
  const h = frame.header, sl = sliceOf(frame, plane);
  if (h.kind === "1d") return plotLayout({ title: { text: title, font: { size: 13 } }, xaxis: { title: "x" } });
  return plotLayout({
    title: { text: title, font: { size: 13 } }, hovermode: "closest", showlegend: false,
    xaxis: { title: sl.axes[1], constrain: "domain", gridcolor: "rgba(0,0,0,0)" },
    yaxis: { title: sl.axes[0], scaleanchor: "x", gridcolor: "rgba(0,0,0,0)" },
    margin: { l: 50, r: 10, t: 34, b: 44 },
  });
}
function volumeTraces(frame, iso, showVolume) {
  const h = frame.header, traces = [];
  if (frame.points && frame.points.length) {
    const p = frame.points, n = p.length / 3, x = new Array(n), y = new Array(n), z = new Array(n);
    for (let i = 0; i < n; i++) { x[i] = p[3 * i]; y[i] = p[3 * i + 1]; z[i] = p[3 * i + 2]; }
    traces.push({ x, y, z, type: "scatter3d", mode: "markers", name: "vortex lines",
      marker: { size: 1.6, color: z, colorscale: "Plasma", opacity: 0.9 }, hovertemplate: "x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>" });
  }
  if (showVolume && frame.volume) {
    const [nx, ny, nz] = h.volume_shape, [Lx, Ly, Lz] = h.box, X = [], Y = [], Z = [], V = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
      X.push(-Lx / 2 + ((i + 0.5) * Lx) / nx); Y.push(-Ly / 2 + ((j + 0.5) * Ly) / ny); Z.push(-Lz / 2 + ((k + 0.5) * Lz) / nz);
      V.push(frame.volume[(i * ny + j) * nz + k] / 255);
    }
    traces.push({ type: "isosurface", x: X, y: Y, z: Z, value: V, isomin: iso, isomax: 1, surface: { count: 1 },
      opacity: 0.25, colorscale: "Blues", showscale: false, caps: { x: { show: false }, y: { show: false }, z: { show: false } },
      name: `density ≥ ${iso} max`, hoverinfo: "skip" });
  }
  return traces;
}
function volumeLayout(frame) {
  const [Lx, Ly, Lz] = frame.header.box, axis = (t, L) => ({ title: t, range: [-L / 2, L / 2], gridcolor: cssVar("--border"), backgroundcolor: "rgba(0,0,0,0)" });
  return plotLayout({ margin: { l: 0, r: 0, t: 10, b: 0 }, showlegend: false, hovermode: "closest",
    scene: { xaxis: axis("x", Lx), yaxis: axis("y", Ly), zaxis: axis("z", Lz), aspectmode: "data" } });
}

/* ------------------------------------------------------------------ dashboard */
let dashSort = { key: "modified", asc: false };
async function viewDashboard() {
  $("#crumbs").innerHTML = "all runs";
  const app = $("#app");
  app.innerHTML = "";
  const box = el(`<div class="panel"><div class="toolbar"><h2 style="margin:0">Runs</h2>
    <div class="chips"><button id="cmp" disabled>Compare selected</button></div></div>
    <div class="scroll"><table class="runs"><thead></thead><tbody></tbody></table></div>
    <p class="muted" style="margin:10px 0 0">Click a column to sort. Select two or more runs to compare.</p></div>`);
  app.appendChild(box);
  const selected = new Set();
  $("#cmp", box).onclick = () => { location.hash = "compare=" + [...selected].map(encodeURIComponent).join(","); };

  async function refresh() {
    let runs;
    try { runs = await listRuns(); } catch (e) { box.querySelector("tbody").innerHTML = `<tr><td>${esc(e.message)}</td></tr>`; return; }
    if (runs.length === 1 && !EMBED && state.firstLoad) { state.firstLoad = false; location.hash = "run=" + encodeURIComponent(runs[0].name); return; }
    state.firstLoad = false;
    const paramKeys = [...new Set(runs.flatMap((r) => Object.keys((r.meta && r.meta.params) || {})))].slice(0, 6);
    const metricKeys = [...new Set(runs.flatMap((r) => Object.keys((r.status && r.status.metrics) || {})))].slice(0, 4);
    const cols = [
      { key: "sel", label: "" }, { key: "title", label: "Run" }, { key: "state", label: "State" },
      { key: "progress", label: "Progress" }, { key: "modified", label: "Updated" }, { key: "wall", label: "Wall" },
      ...metricKeys.map((k) => ({ key: "m:" + k, label: k })), ...paramKeys.map((k) => ({ key: "p:" + k, label: k })),
    ];
    const value = (r, key) => {
      const st = r.status || {}, meta = r.meta || {};
      if (key === "title") return meta.title || st.title || r.name;
      if (key === "state") return runState(r);
      if (key === "progress") return st.fraction ?? (st.state === "finished" ? 1 : null);
      if (key === "modified") return r.modified;
      if (key === "wall") return st.wall_s;
      if (key.startsWith("m:")) return (st.metrics || {})[key.slice(2)];
      if (key.startsWith("p:")) return (meta.params || {})[key.slice(2)];
      return null;
    };
    runs.sort((a, b) => {
      const va = value(a, dashSort.key), vb = value(b, dashSort.key);
      const c = va === vb ? 0 : va === null || va === undefined ? 1 : vb === null || vb === undefined ? -1 : va < vb ? -1 : 1;
      return dashSort.asc ? c : -c;
    });
    box.querySelector("thead").innerHTML = "<tr>" + cols.map((c) =>
      `<th data-k="${c.key}" class="${dashSort.key === c.key ? "sorted" + (dashSort.asc ? " asc" : "") : ""}">${esc(c.label)}</th>`).join("") + "</tr>";
    box.querySelectorAll("th").forEach((th) => th.onclick = () => {
      const k = th.dataset.k; if (k === "sel") return;
      dashSort = { key: k, asc: dashSort.key === k ? !dashSort.asc : true }; refresh();
    });
    box.querySelector("tbody").innerHTML = runs.map((r) => {
      const st = r.status || {}, s = runState(r), frac = value(r, "progress");
      return `<tr><td><input type="checkbox" data-run="${esc(r.name)}" ${selected.has(r.name) ? "checked" : ""}></td>
        <td><a href="#run=${encodeURIComponent(r.name)}">${esc(value(r, "title"))}</a><div class="muted" style="font-size:12px">${esc(r.name)}</div></td>
        <td><span class="badge ${s}">${s}</span></td>
        <td>${frac === null || frac === undefined ? `<span class="muted">${esc(st.progress_label || "t")} ${fmt(st.t)}</span>` :
          `<span class="mini"><div style="width:${(100 * frac).toFixed(1)}%"></div></span> ${(100 * frac).toFixed(0)}%`}</td>
        <td class="muted">${r.modified ? ago(r.modified) : "–"}</td><td>${hms(st.wall_s)}</td>
        ${metricKeys.map((k) => `<td>${fmt((st.metrics || {})[k])}</td>`).join("")}
        ${paramKeys.map((k) => `<td>${fmt(((r.meta || {}).params || {})[k])}</td>`).join("")}</tr>`;
    }).join("");
    box.querySelectorAll("input[type=checkbox]").forEach((cb) => cb.onchange = () => {
      cb.checked ? selected.add(cb.dataset.run) : selected.delete(cb.dataset.run);
      $("#cmp", box).disabled = selected.size < 2;
    });
  }
  await refresh();
  if (!EMBED) state.timers.push(setInterval(refresh, POLL_MS * 2));
}

/* ------------------------------------------------------------------ run view */
async function viewRun(name) {
  const app = $("#app");
  app.innerHTML = "";
  const ui = { view: "density", cmap: "Viridis", log: false, vortices: null, follow: true, frame: -1, iso: 0.3, volume: null, plane: "xy" };
  const page = el(`<div>
    <div class="panel"><div class="runhead"><h1 id="title"></h1><span id="badge" class="badge"></span>
      <span id="prog" class="muted"></span></div>
      <div class="progress"><div id="bar"></div></div><div id="stage" class="stage" hidden></div></div>
    <div class="grid-run">
      <div class="panel">
        <div class="controls">
          <div class="tabs" id="tabs"><button data-v="density" class="on">Density</button><button data-v="phase">Phase</button><button data-v="3d" hidden>3D</button></div>
          <select id="plane" title="Slice through the box centre" hidden><option value="xy">xy plane</option><option value="xz">xz plane</option><option value="yz">yz plane</option></select>
          <select id="cmap" title="Colormap"><option>Viridis</option><option>Inferno</option><option>Cividis</option><option>Plasma</option><option>Greys</option><option>Hot</option></select>
          <label class="check"><input type="checkbox" id="log"> log scale</label>
          <label class="check" id="vortwrap"><input type="checkbox" id="vort" checked> vortices</label>
          <label class="check" id="volwrap" hidden title="Surface at a fraction of the maximum density, from a coarse (<= 48^3) volume: shows cloud shapes; vortex cores are usually finer than this resolution, use the points"><input type="checkbox" id="vol" checked> isosurface</label>
          <label class="check" id="isowrap" hidden>level <input type="range" id="iso" min="0.05" max="0.95" step="0.05" value="0.3" style="width:90px"></label>
        </div>
        <div id="field" class="plot"></div>
        <div class="timeline">
          <button id="play" class="ghost" title="Play / pause">▶</button>
          <input type="range" id="slider" min="0" max="0" value="0">
          <label class="check"><input type="checkbox" id="follow" checked> follow latest</label>
          <span id="frameinfo" class="muted"></span>
        </div>
      </div>
      <div class="side">
        <div class="panel"><h2>Current step</h2><div id="current"></div></div>
        <div class="panel"><h2>Last checkpoint</h2><div id="latest"></div></div>
        <div class="panel"><h2>Parameters</h2><div id="params"></div></div>
        <div class="panel"><h2>Share</h2><p class="muted" style="margin:0">Export this run:<br>
          <code>uv run python -m dipgpe.live export ${esc(name === "." ? "<run dir>" : "results/" + name)} --html run.html --video run.gif --png run.png</code></p></div>
      </div>
    </div>
    <div class="charts" id="charts"></div></div>`);
  app.appendChild(page);
  let run = await loadRun(name);
  $("#crumbs").innerHTML = `<a href="#runs">all runs</a> / ${esc(name)}`;

  const tabs = page.querySelectorAll("#tabs button");
  tabs.forEach((b) => b.onclick = () => { ui.view = b.dataset.v; tabs.forEach((x) => x.classList.toggle("on", x === b)); syncControls(); drawField(); });
  $("#cmap", page).onchange = (e) => { ui.cmap = e.target.value; drawField(); };
  $("#plane", page).onchange = (e) => { ui.plane = e.target.value; drawField(); };
  $("#log", page).onchange = (e) => { ui.log = e.target.checked; drawField(); };
  $("#vort", page).onchange = (e) => { ui.vortices = e.target.checked; drawField(); };
  $("#vol", page).onchange = (e) => { ui.volume = e.target.checked; drawField(); };
  $("#iso", page).onchange = (e) => { ui.iso = +e.target.value; drawField(); };
  $("#follow", page).onchange = (e) => { ui.follow = e.target.checked; if (ui.follow) { ui.frame = -1; drawField(); } };
  $("#slider", page).oninput = (e) => { ui.follow = false; $("#follow", page).checked = false; ui.frame = +e.target.value; drawField(); };
  $("#play", page).onclick = () => {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("#play", page).textContent = "▶"; return; }
    ui.follow = false; $("#follow", page).checked = false;
    if (ui.frame < 0 || ui.frame >= run.frames.length - 1) ui.frame = 0;
    $("#play", page).textContent = "❚❚";
    state.playing = setInterval(async () => {
      if (ui.frame >= run.frames.length - 1) { clearInterval(state.playing); state.playing = null; $("#play", page).textContent = "▶"; return; }
      ui.frame += 1; await drawField();
    }, 350);
  };

  function syncControls() {
    const kind = (run.frames[0] || {}).kind;
    page.querySelector('#tabs [data-v="3d"]').hidden = kind !== "3d";
    const is3d = ui.view === "3d";
    $("#cmap", page).hidden = is3d || ui.view === "phase";
    $("#plane", page).hidden = kind !== "3d" || is3d;
    $("#log", page).parentElement.hidden = is3d || ui.view === "phase";
    $("#vortwrap", page).hidden = kind !== "2d" || is3d;
    $("#volwrap", page).hidden = !is3d; $("#isowrap", page).hidden = !is3d;
  }

  let drawing = false;
  async function drawField() {
    if (drawing) return; drawing = true;
    try {
      const n = run.frames.length, slider = $("#slider", page);
      slider.max = Math.max(0, n - 1);
      let header = null, label = "";
      if (ui.follow) {
        const latest = runState(run) === "running" ? await getJSON(runPath(name, "frames/latest.json")).catch(() => null) : null;
        const last = n ? run.frames[n - 1] : null;
        header = latest && (!last || (latest.t >= last.t)) ? latest : last;
        if (header === latest && latest) label = `latest step${latest.stage ? " · " + latest.stage : ""}`;
        slider.value = Math.max(0, n - 1);
      } else if (n) {
        ui.frame = Math.min(Math.max(ui.frame, 0), n - 1); header = run.frames[ui.frame]; slider.value = ui.frame;
      }
      const div = $("#field", page);
      if (!header) { div.innerHTML = `<p class="muted pad">No field stored yet.</p>`; $("#frameinfo", page).textContent = ""; return; }
      const frame = await loadFrame(name, header);
      if (ui.volume === null) {            // default: isosurface off when vortex points are shown
        ui.volume = !frame.points;
        $("#vol", page).checked = ui.volume;
      }
      if (ui.vortices === null) {          // default: markers on unless they would hide the field
        ui.vortices = !(frame.charges && frame.charges.length > 2000);
        $("#vort", page).checked = ui.vortices;
      }
      const plabel = run.status.progress_label || run.meta.progress_label || "t";
      $("#frameinfo", page).textContent = `${label || `frame ${header.i + 1} / ${n}`} · ${plabel} = ${fmt(header.t)}` + (header.stage && !label ? ` · ${header.stage}` : "");
      syncControls();
      if (ui.view === "3d" && header.kind === "3d") Plotly.react(div, volumeTraces(frame, ui.iso, ui.volume), volumeLayout(frame), PLOT_CONFIG);
      else {
        const sl = sliceOf(frame, ui.plane);
        const title = header.kind === "3d" ? `${ui.view === "phase" ? "arg ψ" : "|ψ|²"} in the ${sl.axes.join("")} plane at ${sl.at ? sl.at[0] + " = " + fmt(sl.at[1]) : "z = " + fmt(header.slice_z)}` : "";
        Plotly.react(div, fieldTraces(frame, ui), fieldLayout(frame, title, ui.plane), PLOT_CONFIG);
      }
    } catch (e) { $("#field", page).innerHTML = `<p class="muted pad">${esc(e.message)}</p>`; }
    finally { drawing = false; }
  }

  const chartState = {};
  function drawCharts() {
    const box = $("#charts", page);
    const keys = [...new Set(run.history.flatMap((r) => Object.keys(r)))].filter((k) => k !== "t" && k !== "wall_s");
    const plabel = run.status.progress_label || run.meta.progress_label || "t";
    for (const k of keys) {
      let card = box.querySelector(`[data-metric="${CSS.escape(k)}"]`);
      if (!card) {
        card = el(`<div class="panel" data-metric="${esc(k)}"><div class="toolbar" style="margin-bottom:4px"><h2 style="margin:0">${esc(k)}</h2>
          <label class="check"><input type="checkbox"> log y</label></div><div class="plot small"></div></div>`);
        box.appendChild(card);
        card.querySelector("input").onchange = (e) => { chartState[k] = e.target.checked; drawCharts(); };
      }
      const logy = !!chartState[k];
      const rows = run.history.filter((r) => typeof r[k] === "number" && (!logy || r[k] > 0));
      Plotly.react(card.querySelector(".plot"), [{ x: rows.map((r) => r.t), y: rows.map((r) => r[k]), type: "scatter", mode: "lines+markers",
        marker: { size: 4 }, line: { color: cssVar("--accent"), width: 2 }, name: k }],
        plotLayout({ margin: { l: 60, r: 10, t: 8, b: 40 }, showlegend: false, xaxis: { title: plabel, gridcolor: cssVar("--border") },
          yaxis: { type: logy ? "log" : "linear", gridcolor: cssVar("--border"), exponentformat: "e" } }), PLOT_CONFIG);
    }
  }

  function drawHeader() {
    const st = run.status, meta = run.meta, s = runState(run);
    $("#title", page).textContent = meta.title || st.title || name;
    const badge = $("#badge", page); badge.textContent = s; badge.className = "badge " + s;
    const plabel = st.progress_label || meta.progress_label || "t";
    let text = `${plabel} = ${fmt(st.t)}${st.total_time ? " / " + fmt(st.total_time) : ""}`;
    if (st.fraction !== undefined && st.fraction !== null) text += ` · ${(100 * st.fraction).toFixed(1)}%`;
    if (s === "running" && st.eta_s !== undefined) text += ` · ETA ${hms(st.eta_s)}`;
    text += ` · wall ${hms(st.wall_s)}` + (meta.git ? ` · git ${meta.git}` : "");
    $("#prog", page).textContent = text;
    $("#bar", page).style.width = `${100 * (st.fraction ?? (s === "finished" ? 1 : 0))}%`;
    const stage = $("#stage", page); stage.hidden = !st.stage; stage.textContent = st.stage || "";
    $("#current", page).innerHTML = kv(st.current);
    $("#latest", page).innerHTML = kv(st.metrics);
    $("#params", page).innerHTML = kv(meta.params);
  }

  drawHeader(); drawCharts(); await drawField();
  if (!EMBED) state.timers.push(setInterval(async () => {
    if (runState(run) === "finished" && run.done) return;
    const fresh = await loadRun(name).catch(() => null);
    if (!fresh) return;
    const grew = fresh.frames.length !== run.frames.length || fresh.history.length !== run.history.length;
    fresh.done = runState(fresh) === "finished";
    run = fresh; drawHeader();
    if (grew) drawCharts();
    if (ui.follow) drawField();
  }, POLL_MS));
}

/* ------------------------------------------------------------------ compare view */
async function viewCompare(names) {
  const app = $("#app");
  app.innerHTML = "";
  $("#crumbs").innerHTML = `<a href="#runs">all runs</a> / compare ${names.length} runs`;
  const runs = await Promise.all(names.map(loadRun));
  const colors = ["#2f6fdb", "#e4572e", "#29a36a", "#a259c4", "#f2a541", "#17bebb"];
  const titleOf = (r) => r.meta.title || r.status.title || r.name;
  const metrics = [...new Set(runs.flatMap((r) => r.history.flatMap((h) => Object.keys(h))))].filter((k) => k !== "t" && k !== "wall_s");
  const page = el(`<div>
    <div class="panel"><div class="toolbar"><h2 style="margin:0">Metric</h2>
      <div class="chips">${runs.map((r, i) => `<span class="chip" style="border-color:${colors[i % 6]};color:${colors[i % 6]}">${esc(titleOf(r))}</span>`).join("")}</div></div>
      <div class="controls"><select id="metric">${metrics.map((m) => `<option>${esc(m)}</option>`).join("")}</select>
      <label class="check"><input type="checkbox" id="clog"> log y</label></div>
      <div id="overlay" class="plot small"></div></div>
    <div class="panel" style="margin-top:14px"><div class="toolbar"><h2 style="margin:0">Fields (time-synced)</h2>
      <div class="controls" style="margin:0"><div class="tabs" id="ctabs"><button data-v="density" class="on">Density</button><button data-v="phase">Phase</button></div>
      <label class="check"><input type="checkbox" id="cflog"> log scale</label></div></div>
      <div class="timeline" style="margin:0 0 8px"><span class="muted" id="tlabel"></span><input type="range" id="tslider" min="0" max="1000" value="1000"></div>
      <div class="compare-fields" id="cfields"></div></div>
    <div class="panel" style="margin-top:14px"><h2>Parameters and results</h2><div class="scroll" id="ctable"></div></div></div>`);
  app.appendChild(page);
  const ui = { metric: metrics[0], log: false, view: "density", flog: false };

  function drawOverlay() {
    const traces = runs.map((r, i) => {
      const rows = r.history.filter((h) => typeof h[ui.metric] === "number" && (!ui.log || h[ui.metric] > 0));
      return { x: rows.map((h) => h.t), y: rows.map((h) => h[ui.metric]), name: titleOf(r), type: "scatter", mode: "lines+markers",
        marker: { size: 4 }, line: { color: colors[i % 6], width: 2 } };
    });
    const plabel = runs[0].meta.progress_label || runs[0].status.progress_label || "t";
    Plotly.react($("#overlay", page), traces, plotLayout({ xaxis: { title: plabel, gridcolor: cssVar("--border") },
      yaxis: { title: ui.metric, type: ui.log ? "log" : "linear", gridcolor: cssVar("--border"), exponentformat: "e" }, margin: { l: 64, r: 10, t: 10, b: 70 } }), PLOT_CONFIG);
  }
  $("#metric", page).onchange = (e) => { ui.metric = e.target.value; drawOverlay(); };
  $("#clog", page).onchange = (e) => { ui.log = e.target.checked; drawOverlay(); };

  const withFrames = runs.filter((r) => r.frames.length);
  const tmax = Math.max(0, ...withFrames.map((r) => r.frames[r.frames.length - 1].t));
  const box = $("#cfields", page);
  withFrames.slice(0, 4).forEach((r) => box.appendChild(el(`<div><div class="muted" style="margin-bottom:4px">${esc(titleOf(r))}</div><div class="plot compare" data-run="${esc(r.name)}"></div></div>`)));
  if (!withFrames.length) box.innerHTML = `<p class="muted">None of these runs stored fields.</p>`;
  async function drawFields() {
    const target = (+$("#tslider", page).value / 1000) * tmax;
    $("#tlabel", page).textContent = `t ≈ ${fmt(target)}`;
    for (const r of withFrames.slice(0, 4)) {
      let best = r.frames[0];
      for (const f of r.frames) if (Math.abs(f.t - target) < Math.abs(best.t - target)) best = f;
      const frame = await loadFrame(r.name, best);
      const div = box.querySelector(`[data-run="${CSS.escape(r.name)}"]`);
      Plotly.react(div, fieldTraces(frame, { view: ui.view, cmap: "Viridis", log: ui.flog, vortices: false }),
        fieldLayout(frame, `t = ${fmt(best.t)}`), PLOT_CONFIG);
    }
  }
  $("#tslider", page).oninput = drawFields;
  page.querySelectorAll("#ctabs button").forEach((b) => b.onclick = () => {
    ui.view = b.dataset.v; page.querySelectorAll("#ctabs button").forEach((x) => x.classList.toggle("on", x === b)); drawFields();
  });
  $("#cflog", page).onchange = (e) => { ui.flog = e.target.checked; drawFields(); };

  let sort = { key: null, asc: true };
  function drawTable() {
    const params = [...new Set(runs.flatMap((r) => Object.keys(r.meta.params || {})))];
    const finals = [...new Set(runs.flatMap((r) => Object.keys(r.status.metrics || {})))];
    const cols = ["run", "state", "wall", ...params.map((p) => "p:" + p), ...finals.map((m) => "m:" + m)];
    const val = (r, c) => c === "run" ? titleOf(r) : c === "state" ? runState(r) : c === "wall" ? r.status.wall_s
      : c.startsWith("p:") ? (r.meta.params || {})[c.slice(2)] : (r.status.metrics || {})[c.slice(2)];
    const rows = [...runs];
    if (sort.key) rows.sort((a, b) => { const x = val(a, sort.key), y = val(b, sort.key); const c = x === y ? 0 : x < y ? -1 : 1; return sort.asc ? c : -c; });
    const label = (c) => c.startsWith("p:") || c.startsWith("m:") ? c.slice(2) : c;
    $("#ctable", page).innerHTML = `<table class="runs"><thead><tr>${cols.map((c) => `<th data-c="${esc(c)}" class="${sort.key === c ? "sorted" + (sort.asc ? " asc" : "") : ""}">${esc(label(c))}${c.startsWith("m:") ? " (final)" : ""}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${c === "wall" ? hms(val(r, c)) : fmt(val(r, c))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    $("#ctable", page).querySelectorAll("th").forEach((th) => th.onclick = () => { const c = th.dataset.c; sort = { key: c, asc: sort.key === c ? !sort.asc : true }; drawTable(); });
  }
  drawOverlay(); drawTable(); await drawFields();
}

/* ------------------------------------------------------------------ routing */
async function route() {
  clearTimers();
  const h = decodeURIComponent(location.hash.slice(1));
  try {
    if (h.startsWith("run=")) await viewRun(h.slice(4));
    else if (h.startsWith("compare=")) await viewCompare(h.slice(8).split(",").filter(Boolean));
    else await viewDashboard();
  } catch (e) { $("#app").innerHTML = `<div class="panel"><p>${esc(e.message)}</p></div>`; }
}
state.firstLoad = true;
if (typeof Plotly === "undefined") {
  window.Plotly = { react: (div) => { div.innerHTML = `<p class="muted pad">Plots need Plotly from cdnjs.cloudflare.com; it could not be loaded (offline?).</p>`; } };
}
window.addEventListener("hashchange", route);
if (EMBED && EMBED.start && !location.hash) location.hash = EMBED.start;
route();

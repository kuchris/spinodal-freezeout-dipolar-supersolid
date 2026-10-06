"""dipgpe.live: frame encoding, run files, the HTTP server and exports."""

import json
import math
import urllib.request

import numpy as np
import pytest
import torch

import dipgpe
from dipgpe.fields import imprint_vortex_rings, imprint_vortices
from dipgpe.live import LiveRun, decode_frame, encode_frame
from dipgpe.vortices import find_vortex_points, find_vortices
from util import C64


def write_frame(tmp_path, header, files):
    header = dict(header, files={})
    for key, data in files.items():
        (tmp_path / f"x_{key}.bin").write_bytes(data)
        header["files"][key] = f"x_{key}.bin"
    return decode_frame(tmp_path, header)


def test_frame_round_trip_2d_with_vortices(tmp_path):
    grid = dipgpe.Grid((512, 256), (32.0, 16.0), dtype=C64)
    psi = imprint_vortices(grid, torch.ones(grid.shape, dtype=C64), [(3.1, 0.2), (-3.9, 0.3)], [1, -1], 1.0)
    vort = find_vortices(grid, psi)
    header, files = encode_frame(grid, psi, vort, frame_size=128)
    out = write_frame(tmp_path, header, files)
    assert header["shape"] == [128, 64] and header["kind"] == "2d"
    pooled = torch.nn.functional.avg_pool2d((psi.abs() ** 2)[None, None], 4)[0, 0].sqrt().numpy()
    assert np.abs(out["amp"] - pooled).max() <= header["amp_max"] / 65535
    expected_phase = torch.angle(psi[::4, ::4]).numpy()
    wrapped = np.angle(np.exp(1j * (out["phase"] - expected_phase)))
    assert np.abs(wrapped).max() <= math.pi / 255 + 1e-6
    assert np.allclose(out["points"], vort[0].numpy(), atol=1e-5) and list(out["charges"]) == vort[1].tolist()


def test_frame_round_trip_3d_and_1d(tmp_path):
    grid = dipgpe.Grid((48, 48, 48), 24.0, dtype=C64)
    psi = imprint_vortex_rings(grid, torch.ones(grid.shape, dtype=C64), [{"center": (0.1, 0.1, 0.0), "radius": 6.0}], 1.0)
    points = find_vortex_points(grid, psi)[0]
    header, files = encode_frame(grid, psi, (points,))
    out = write_frame(tmp_path, header, files)
    assert header["kind"] == "3d" and header["volume_shape"] == [48, 48, 48]
    assert np.abs(out["density_volume"] - (psi.abs() ** 2).numpy()).max() <= header["volume_max"] / 255 / 2 + 1e-6
    assert out["points"].shape == (len(points), 3)
    line = dipgpe.Grid(1000, 20.0, dtype=C64)
    h1, f1 = encode_frame(line, dipgpe.gaussian(line, 0.0, 1.0))
    assert h1["shape"] == [250] and write_frame(tmp_path, h1, f1)["amp"].shape == (250,)


def test_frame_3d_mid_planes(tmp_path):
    grid = dipgpe.Grid((32, 24, 40), (16.0, 12.0, 30.0), dtype=C64)
    x, y, z = grid.x64
    psi = (torch.exp(-(x * x + y * y) / 4) * (1.5 + torch.cos(2 * math.pi * z / 10) + 0.1 * x)
           * torch.exp(1j * 0.3 * z)).to(C64)
    header, files = encode_frame(grid, psi)
    out = write_frame(tmp_path, header, files)
    planes = header["planes"]
    assert planes["xz"]["axes"] == ["x", "z"] and planes["yz"]["axes"] == ["y", "z"]
    assert planes["xz"]["extent"] == [-8.0, 8.0, -15.0, 15.0] and planes["yz"]["shape"] == [24, 40]
    for key, slab in (("xz", psi[:, 12, :]), ("yz", psi[16, :, :])):
        assert planes[key]["at"][1] == pytest.approx(0.0, abs=1e-12)
        assert np.abs(out[f"amp_{key}"] - slab.abs().numpy()).max() <= planes[key]["amp_max"] / 65535 + 1e-6
        wrapped = np.angle(np.exp(1j * (out[f"phase_{key}"] - torch.angle(slab).numpy())))
        assert np.abs(wrapped).max() <= math.pi / 255 + 1e-6
    assert np.allclose(out["amp"], psi[:, :, 20].abs().numpy(), atol=header["amp_max"] / 65535 + 1e-6)


def test_live_run_files_api_and_viewer(tmp_path):
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C64)
    psi = dipgpe.gaussian(grid, 0.0, 1.5)
    live = LiveRun(tmp_path / "runA", "test run", total_time=10.0, port=0, params={"N": 64, "g": 1.5})
    live.stage("warming up")
    live.report({"mu": 1.0}, grid, psi)
    for t in (0.0, 5.0):
        live.checkpoint(t, {"energy_rel": 1e-7 * t}, grid, psi)
    run = tmp_path / "runA"
    meta = json.loads((run / "meta.json").read_text())
    status = json.loads((run / "status.json").read_text())
    assert meta["params"] == {"N": 64, "g": 1.5} and meta["format"] == 2
    assert status["fraction"] == 0.5 and status["frames"] == 2 and status["stage"] == "warming up"
    assert len((run / "frames" / "index.jsonl").read_text().splitlines()) == 2
    assert (run / "frames" / "latest.json").exists()
    base = f"http://127.0.0.1:{live.server.server_address[1]}"
    runs = json.loads(urllib.request.urlopen(base + "/api/runs").read())
    assert [r["name"] for r in runs] == ["runA"] and runs[0]["meta"]["title"] == "test run"
    assert b"DipGPE" in urllib.request.urlopen(base + "/").read()
    assert b"decodeFrame" in urllib.request.urlopen(base + "/_viewer/app.js").read()
    assert urllib.request.urlopen(base + "/runA/frames/index.jsonl").status == 200
    live.close()
    assert json.loads((run / "status.json").read_text())["state"] == "finished"
    live.server.shutdown()


def test_ground_state_reports_to_live(tmp_path):
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C64)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0))
    live = LiveRun(tmp_path / "gs", "ground state", total_time=1, progress_label="task")
    live.stage("ground state")
    psi, info = dipgpe.ground_state(model, dipgpe.normalize(grid, dipgpe.gaussian(grid, 0.0, 1.5)), live=live)
    status = json.loads((tmp_path / "gs" / "status.json").read_text())
    assert abs(status["current"]["mu"] - float(info["mu"])) < 1e-6 and status["progress_label"] == "task"
    live.close()


def test_exports(tmp_path):
    pytest.importorskip("matplotlib")
    from dipgpe.live_export import export_html, export_png, export_video
    grid = dipgpe.Grid((64, 64), 16.0, dtype=C64)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(20.0))
    psi = dipgpe.translate(grid, dipgpe.gaussian(grid, 0.0, 1.2), (1.0, 0.0))
    live = LiveRun(tmp_path / "osc", "oscillation", total_time=1.0)
    stepper = dipgpe.SplitStep(model, 0.01)
    for _ in range(5):
        live.checkpoint(stepper.t, {"x": float(dipgpe.center_of_mass(grid, psi)[0])}, grid, psi)
        psi = stepper.evolve(psi, 20)
    live.close()
    html = export_html(tmp_path / "osc", tmp_path / "osc.html").read_text(encoding="utf-8")
    assert "window.QS_EMBED" in html and "osc/frames/f00004_amp.bin" in html and "_viewer/" not in html
    assert export_video(tmp_path / "osc", tmp_path / "osc.gif").stat().st_size > 1000
    assert export_png(tmp_path / "osc", tmp_path / "osc.png").stat().st_size > 1000


def test_exports_3d_plane(tmp_path):
    pytest.importorskip("matplotlib")
    from dipgpe.live_export import export_png, export_video
    grid = dipgpe.Grid((24, 24, 32), (12.0, 12.0, 24.0), dtype=C64)
    x, y, z = grid.x64
    live = LiveRun(tmp_path / "tube", "tube", total_time=2.0)
    for t in (1.0, 2.0):
        psi = (torch.exp(-(x * x + y * y) / 2) * (1 + 0.5 * t * torch.cos(2 * math.pi * z / 8))).to(C64)
        live.checkpoint(t, {"t2": t * t}, grid, psi)
    live.close()
    assert export_video(tmp_path / "tube", tmp_path / "tube.gif", plane="xz").stat().st_size > 1000
    assert export_png(tmp_path / "tube", tmp_path / "tube.png", plane="yz").stat().st_size > 1000

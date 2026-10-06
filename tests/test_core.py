"""Grid, field helpers, snapshot I/O, quick-look and CPU/CUDA agreement."""

import math

import pytest
import torch

import dipgpe
from dipgpe.io import load_snapshot, save_snapshot
from util import C64, C128, rel_l2


def test_grid_coordinates_and_wavenumbers():
    grid = dipgpe.Grid((8, 4), (2.0, 4.0), dtype=C128)
    assert grid.dx == (0.25, 1.0) and grid.dV == 0.25
    assert float(grid.x[0].min()) == -1.0 and float(grid.x[0].max()) == 0.75
    assert grid.x[0].shape == (8, 1) and grid.k[1].shape == (1, 4)
    assert torch.allclose(grid.k2, grid.k[0] ** 2 + grid.k[1] ** 2)


def test_gaussian_is_normalized_and_fft_round_trips():
    grid = dipgpe.Grid((64, 64, 64), 16.0, dtype=C128)
    psi = dipgpe.gaussian(grid, (0.5, -1.0, 0.0), 1.0, (1.0, 0.0, 2.0))
    assert abs(float(dipgpe.norm(grid, psi)) - 1) < 1e-10
    assert rel_l2(grid.ifft(grid.fft(psi)), psi) < 1e-14


def test_translate_is_exact_for_band_limited_fields():
    grid = dipgpe.Grid((96, 96), 24.0, dtype=C128)
    moved = dipgpe.translate(grid, dipgpe.gaussian(grid, (0.0, 0.0), 1.0), (0.37, -1.2))
    assert rel_l2(moved, dipgpe.gaussian(grid, (0.37, -1.2), 1.0)) < 1e-12


def test_batch_axis_evolves_independently():
    grid = dipgpe.Grid(128, 20.0, dtype=C128)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(5.0))
    a, b = dipgpe.gaussian(grid, -1.0, 1.0), dipgpe.gaussian(grid, 2.0, 0.7, 1.0)
    stepper = dipgpe.SplitStep(model, 0.01)
    batched = stepper.evolve(torch.stack([a, b]), 50)
    assert rel_l2(batched[0], stepper.evolve(a, 50)) < 1e-13
    assert rel_l2(batched[1], stepper.evolve(b, 50)) < 1e-13
    assert dipgpe.norm(grid, batched).shape == (2,)


def test_snapshot_round_trip(tmp_path):
    grid = dipgpe.Grid((32, 16), (8.0, 4.0), dtype=C64)
    psi = dipgpe.gaussian(grid, 0.0, 1.0)
    path = save_snapshot(tmp_path / "snap.npz", grid, psi, 1.25, {"g": 10.0})
    grid2, psi2, meta = load_snapshot(path)
    assert grid2.shape == grid.shape and grid2.length == grid.length and grid2.dtype == C64
    assert torch.equal(psi2, psi)
    assert meta["t"] == 1.25 and meta["params"] == {"g": 10.0}
    assert {"python", "torch", "numpy"} <= set(meta["versions"])


def test_quicklook_renders(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from dipgpe.quicklook import show

    for shape in ((64,), (32, 32), (16, 16, 16)):
        grid = dipgpe.Grid(shape, 10.0, dtype=C64)
        show(grid, dipgpe.gaussian(grid, 0.0, 1.0), path=tmp_path / f"q{len(shape)}.png")
    assert len(list(tmp_path.glob("*.png"))) == 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_and_cuda_agree():
    results = []
    for device in ("cpu", "cuda"):
        grid = dipgpe.Grid((64, 64), 16.0, dtype=C128, device=device)
        model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(10.0))
        psi = dipgpe.gaussian(grid, (0.7, -0.3), 0.8, (0.5, 0.2))
        results.append(dipgpe.SplitStep(model, 0.01, dipgpe.YOSHIDA4).evolve(psi, 100).cpu())
    assert rel_l2(results[1], results[0]) < 1e-12

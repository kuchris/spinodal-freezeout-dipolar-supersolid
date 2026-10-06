"""3D vortex rings: periodic imprint, line detection, and the propagation-speed gate.

Units g = n0 = 1 (hbar = m = 1). Literature (Roberts & Grant 1971; as used by
Krstulovic & Brachet, arXiv:1006.4315): U = (hbar / 2 m R) [ln(8 R / xi) - 0.615]
with xi = (hbar^2 / 2 m g n0)^(1/2) = 1/sqrt(2) here, i.e.
    2 R U = ln(8 R) + ln(sqrt 2) - 0.615 = ln(8 R) - 0.268.
The formula is asymptotic in R / xi. The ring radius relaxes by ~1-3% after the
imprint (sound emission), so U is compared at the radius measured over the fit
window. Measured (dx = 0.5): U / U_theory = 1.0168, 1.0097, 1.0068 for
R = 8, 16, 32; dx = 0.25 changes R = 8 by 0.08%. The raw speed does not depend on
the box size (L / R = 4..8), so no periodic correction is applied.
"""

import math

import torch

import dipgpe
from dipgpe.fields import imprint_vortex_rings
from dipgpe.vortices import find_vortex_points
from util import C64, C128, DEVICE


def u_theory(R):
    return (math.log(8 * math.sqrt(2) * R) - 0.615) / (2 * R)


def ring_field(R, L, dx=0.5, dtype=C64, center=None, sign=1):
    N = int(round(L / dx))
    grid = dipgpe.Grid((N, N, N), L, dtype=dtype, device=DEVICE)
    center = center or (0.1, -0.1, -L / 4 + 0.05)
    psi = imprint_vortex_rings(grid, torch.ones(grid.shape, dtype=dtype, device=DEVICE),
                               [{"center": center, "radius": R, "sign": sign}], 1.0)
    return grid, psi, center


def ring_speed(R, L, sign=1, chunks=30):
    grid, psi, c = ring_field(R, L, sign=sign)
    T = min(0.4 * L / u_theory(R), 60.0)
    n = math.ceil(T / chunks / dipgpe.max_stable_dt(grid))
    stepper = dipgpe.SplitStep(dipgpe.GPE(grid, dipgpe.Contact(1.0)), T / chunks / n, keep_norm=True)
    times, zs, radii = [], [], []
    for _ in range(chunks):
        psi = stepper.evolve(psi, n)
        points, _, _ = find_vortex_points(grid, psi)
        points = points.cpu()
        times.append(stepper.t)
        zs.append(float(points[:, 2].mean()))
        radii.append(float(((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2).sqrt().mean()))
    t, z, r = torch.tensor(times), torch.tensor(zs), torch.tensor(radii)
    for i in range(1, len(z)):                       # unwrap across the periodic boundary
        while z[i] - z[i - 1] < -L / 2:
            z[i:] += L
        while z[i] - z[i - 1] > L / 2:
            z[i:] -= L
    late = t >= T / 4
    A = torch.stack([t[late], torch.ones(int(late.sum()), dtype=t.dtype)], 1)
    U = torch.linalg.lstsq(A, z[late, None]).solution[0, 0].item()
    return U, float(r[late].mean())


def test_imprinted_ring_is_smooth_and_detected():
    R = 10.0
    grid, psi, c = ring_field(R, 64.0, dtype=C128)
    power = grid.fft(psi).abs() ** 2
    assert float(power[grid.k2.sqrt() > 0.7 * math.pi / grid.dx[0]].sum() / power.sum()) < 1e-5
    points, axes, charges = find_vortex_points(grid, psi)
    points = points.cpu()
    radius = ((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2).sqrt()
    assert len(points) > 100 and set(axes.tolist()) == {0, 1}      # pierces x- and y-normal planes
    assert abs(float(radius.mean()) - R) < 0.05 and float(radius.std()) < 0.1
    assert abs(float(points[:, 2].mean()) - c[2]) < 0.1


def test_ring_speed_matches_roberts_grant():
    deviations = []
    for R, L in ((8.0, 48.0), (16.0, 96.0)):
        U, R_fit = ring_speed(R, L)
        deviations.append(U / u_theory(R_fit) - 1)
    assert all(abs(d) < 0.025 for d in deviations), deviations
    assert abs(deviations[1]) < abs(deviations[0])               # asymptotic in R / xi


def test_ring_direction_follows_sign():
    U_plus, _ = ring_speed(6.0, 36.0, sign=1, chunks=10)
    U_minus, _ = ring_speed(6.0, 36.0, sign=-1, chunks=10)
    assert U_plus > 0.2 and abs(U_minus + U_plus) < 0.01 * U_plus

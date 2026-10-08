"""Unit tests of the analysis added for paper C1 v2 (Appendices E and F).

- tube_coexistence.coexistence: the (mu, P) crossing must return the double
  tangent of two branches whose coexisting densities are known exactly
  (two parabolas of equal curvature and equal minimum, tilted by a common
  linear term, which leaves the double tangent unchanged).
- tube_coexistence.optimal_crystal: the parabola in the cell length must return
  the minimum of an exactly quadratic e(L) and interpolate mu to it.
- tube_thermal_shift.DampedCosineFit: recovers frequency and damping of an
  exact damped cosine to the grid resolution.
- tube_thermal_shift.median: the convention of tube_ramp_errors.py (upper
  middle for an even count), with non-crossing realizations counted as latest.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import tube_coexistence as tc  # noqa: E402
import tube_thermal_shift as ts  # noqa: E402


def branch(n_um, center_um, k, s, e):
    """Rows of a parabolic energy per length f(n_l) = k (n_l - c)^2 + s n_l + e."""
    rows = []
    for n in n_um:
        nl, c = n * tc.PER_L, center_um * tc.PER_L
        f = k * (nl - c) ** 2 + s * nl + e
        mu = 2 * k * (nl - c) + s
        rows.append({"n": n, "mu": mu, "P": mu * nl - f, "e": f / nl})
    return rows


def test_coexistence_recovers_known_double_tangent():
    n = np.arange(6100.0, 6401.0, 10.0)
    U = branch(n, 6280.0, 2e-3, 12.6, -100.0)
    M = branch(n, 6220.0, 2e-3, 12.6, -100.0)
    co = tc.coexistence(U, M)
    assert co["crossing"]
    assert abs(co["n_M"] - 6220.0) < 1e-3 and abs(co["n_U"] - 6280.0) < 1e-3
    assert abs(co["dn"] - 60.0) < 2e-3


def test_coexistence_absent_for_touching_convex_branches():
    """A crystal branch that only touches the uniform one from below at its
    onset and stays convex (positive slope of mu) has no crossing."""
    n = np.arange(6100.0, 6401.0, 10.0)
    U = branch(n, 6250.0, 2e-3, 12.6, -100.0)
    M = [dict(r, P=r["P"] + 1e-3 * (r["mu"] - U[0]["mu"]) ** 2) for r in U]
    assert not tc.coexistence(U, M).get("crossing")


def test_optimal_crystal_parabola_in_cell_length():
    L0, e0, q = 4.31, 8.4, 3e-4
    rows = []
    for n in (6200.0, 6250.0):
        for L in (4.2, 4.3, 4.4):
            rows.append({"n": n, "L": L, "e": e0 + q * (L - L0) ** 2, "mu": 12.6 + 0.1 * (L - L0),
                         "C_line": 0.7, "converged": True})
    out = tc.optimal_crystal(rows, 0.05)
    assert len(out) == 2
    for o in out:
        assert abs(o["L_opt"] - L0) < 1e-9 and abs(o["e"] - e0) < 1e-12 and abs(o["mu"] - 12.6) < 1e-9
        assert not o["edge"]


def test_damped_cosine_fit_recovers_frequency_and_damping():
    tau = np.arange(100) * 0.9425                       # 1 ms steps in units of 1/omega_perp
    fit = ts.DampedCosineFit(tau)
    w, g = fit(np.exp(-0.02 * tau) * np.cos(0.41 * tau))
    assert abs(w - 0.41) <= 0.0005 and abs(g - 0.02) <= 0.002


def test_median_convention_matches_appendix_d():
    assert ts.median(np.array([4.0, 1.0, 3.0, 2.0])) == 3.0       # upper middle of an even count
    assert ts.median(np.array([1.0, 2.0, 3.0])) == 2.0
    assert ts.median(np.array([1.0, math.nan, 2.0])) == 2.0      # never crossing counts as latest

"""Gate: the dipolar tube crystallizes where the literature puts the transition.

164Dy in an infinite tube (omega_perp = 2 pi x 150 Hz, dipoles along y,
a_dd = 130.8 a0, n = 2500 atoms/um, LHY with Q5). At this density the transition
is continuous with a* = a_rot* (Smith, Baillie & Blakie, arXiv:2212.07607).
Literature, quoted verbatim (docs/ISSUES.md P4-3): reduced theory a* ≈ 91.6 a0
(arXiv:2004.12577); full eGPE "approximately 1 a0 to 2 a0" higher.

Measured with the roton-softening rate (benchmarks/tube_supersolid.py): full
eGPE a_rot* = 92.32 a0, converged in dx, transverse box, dt and N_z to 0.01 a0,
and confirmed by crystal ground states (contrast 0.18 at 92.25 a0, 0 at 92.5 a0).
That is 0.72 a0 above the reduced theory, slightly below the literature's
"approximately 1-2 a0" (docs/RESULTS.md discusses this).

The test checks (dx = 0.25 gives the same rates as 0.125):
  * the uniform tube is unstable at the reduced-theory value 91.6 a0 and stable
    2 a0 above it, i.e. the full-eGPE transition lies in (91.6, 93.6) a0;
  * the measured crossing reproduces 92.32 +- 0.05 a0.
"""

import sys
from pathlib import Path

import pytest

from util import DEVICE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from tube_supersolid import softening_rate  # noqa: E402

N_PER_UM, CELL = 2500.0, 4.4        # cell length 4.4 l = 2.82 um, near the roton wavelength


def rate(a_s):
    return softening_rate(a_s, N_PER_UM, CELL, dx=0.25)


@pytest.mark.gpu
@pytest.mark.skipif(DEVICE != "cuda", reason="3D tube states need the GPU for reasonable runtime")
def test_tube_transition_above_reduced_theory_and_within_2_a0():
    assert rate(91.6) < 0 < rate(93.6)
    lo, hi = rate(92.25), rate(92.5)
    crossing = 92.25 + 0.25 * (-lo) / (hi - lo)
    assert abs(crossing - 92.32) < 0.05, crossing

"""Gate: the 164Dy tube transition at n = 6250/um is discontinuous.

Literature (Blakie et al., arXiv:2004.12577, reduced theory, verbatim): "the
transition occurs at a* = 89.6a0 (cf. arot = 88.6a0)", i.e. the crystal becomes
the ground state before the roton softens. benchmarks/tube_first_order.py
measures, in the full eGPE, a* = 90.256 a0 at cell 4.3 l (dx 0.25 and 0.125
agree to 1e-5 a0, transverse box 24 -> 30 l moves it by 0.002 a0), against
a_rot* = 89.83 a0 from roton softening, with contrast 0.71 at a* and a
metastable crystal branch up to 90.375-90.4 a0.

The test follows the crystal branch at the optimal cell (dx = 0.25) and checks:
a* above a_rot* by more than 0.3 a0, a finite contrast at a*, and a crystal
that survives above a* with a higher energy than the uniform tube (hysteresis).
"""

import sys
from pathlib import Path

import pytest

from util import DEVICE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from tube_first_order import analyse, crystal_branch, uniform_energy  # noqa: E402

N_PER_UM, CELL, DX, L_PERP, A_ROT = 6250.0, 4.3, 0.25, 24.0, 89.83


@pytest.mark.gpu
@pytest.mark.skipif(DEVICE != "cuda", reason="3D tube states need the GPU for reasonable runtime")
def test_high_density_tube_transition_is_first_order():
    a_list = [90.0, 90.1, 90.2, 90.25, 90.3, 90.35, 90.45]
    E_uniform = {a: uniform_energy(N_PER_UM, a, DX, L_PERP) for a in a_list}
    out = analyse(crystal_branch(N_PER_UM, CELL, a_list, E_uniform, DX, L_PERP, width=1.0))
    assert out["kind"] == "crossing"
    assert abs(out["a_star"] - 90.256) < 0.01, out["a_star"]
    assert out["a_star"] - A_ROT > 0.3
    assert out["C_at_a_star"] > 0.5
    assert out["metastable_dE_max"] is not None and out["metastable_dE_max"] > 0
    assert out["a_end"] == [90.35, 90.45], out["a_end"]

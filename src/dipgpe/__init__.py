"""dipgpe: uniform-grid pseudo-spectral Gross-Pitaevskii solver."""

from .diagnostics import (angular_momentum, center_of_mass, chemical_potential, density, energy,
                          energy_components, expectation, kinetic_energy, norm,
                          residual)
from .fields import gaussian, normalize, resample, translate
from .grid import Grid
from .model import GPE
from .minimize import ground_state
from .stepper import STRANG, YOSHIDA4, Scheme, SplitStep, max_stable_dt
from .terms import (LHY, Contact, Dipolar, Potential, Rotation, Twist, gaussian_obstacle, harmonic,
                    lhy_coefficient, q5)

__all__ = [
    "Grid", "GPE", "Potential", "Contact", "Rotation", "Twist", "Dipolar", "LHY", "harmonic",
    "gaussian_obstacle", "lhy_coefficient", "q5",
    "SplitStep", "Scheme", "STRANG", "YOSHIDA4", "ground_state", "max_stable_dt",
    "gaussian", "normalize", "resample", "translate",
    "density", "norm", "kinetic_energy", "energy", "energy_components",
    "chemical_potential", "residual", "expectation", "center_of_mass", "angular_momentum",
]


def __getattr__(name):
    # dipgpe.runs is imported on first use, so `python -m dipgpe.runs` does not import it twice
    if name in ("runs", "bdg"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'dipgpe' has no attribute {name!r}")

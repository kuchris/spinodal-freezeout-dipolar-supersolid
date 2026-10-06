"""Conversion between SI units and dipgpe's dimensionless units (hbar = m = 1).

With length unit l and time unit 1/omega (l = sqrt(hbar / (m omega))), energies
are in hbar omega, g = 4 pi a / l for a scattering length a, and a dipolar
interaction has g_dd = 4 pi a_dd / l.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34        # J s
AMU = 1.66053906660e-27       # kg
BOHR = 5.29177210903e-11      # m


class OscillatorUnits:
    """Units of a harmonic oscillator of angular frequency ``omega`` (rad/s)."""

    def __init__(self, mass_amu, omega):
        self.mass = mass_amu * AMU
        self.omega = float(omega)
        self.length = math.sqrt(HBAR / (self.mass * self.omega))   # m
        self.time = 1.0 / self.omega                                # s

    def bohr(self, a_bohr):
        """A length given in Bohr radii, in units of l."""
        return a_bohr * BOHR / self.length

    def per_micron(self, n_per_um):
        """A linear density given in atoms per micrometre, in atoms per l."""
        return n_per_um * self.length * 1e6

    def microns(self, x):
        return x * self.length * 1e6

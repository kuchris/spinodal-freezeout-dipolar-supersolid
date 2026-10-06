"""The GPE model: a grid plus a set of local terms."""

from __future__ import annotations


class GPE:
    """i dpsi/dt = [ -1/2 nabla^2 + sum of terms ] psi.

    Terms are listed in :mod:`dipgpe.terms`. The kinetic term is implicit. Local
    terms provide ``local``; a :class:`dipgpe.terms.Rotation` is applied by the
    stepper inside the kinetic step (at most one per model).
    """

    def __init__(self, grid, *terms):
        self.grid = grid
        self.terms = tuple(terms)
        self.local_terms = tuple(t for t in self.terms if hasattr(t, "local"))
        rotations = [t for t in self.terms if hasattr(t, "omega")]
        if len(rotations) > 1:
            raise ValueError("at most one Rotation term")
        if rotations and grid.ndim < 2:
            raise ValueError("rotation needs a 2D or 3D grid")
        self.rotation = rotations[0].omega if rotations else 0.0

    def local_potential(self, density, t=0.0):
        """Sum of the local terms' potentials U(x, t) for the given density |psi|^2."""
        U = 0.0
        for term in self.local_terms:
            U = U + term.local(density, t)
        return U

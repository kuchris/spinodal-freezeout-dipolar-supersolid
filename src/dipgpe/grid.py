"""Uniform periodic grids in one, two or three dimensions."""

from __future__ import annotations

import math

import torch

REAL_DTYPE = {torch.complex64: torch.float32, torch.complex128: torch.float64}


class Grid:
    """Periodic box [-L/2, L/2) per axis with N equally spaced points.

    Fields keep the spatial axes last; any leading axes (batch, component) are
    free. ``x[i]`` and ``k[i]`` are broadcastable along spatial axis ``i`` in
    the field's real precision; ``x64``/``k64`` hold the same 1D-broadcast axes
    in float64, for building phases and initial fields accurately.
    """

    def __init__(self, shape, length, *, dtype=torch.complex64, device="cpu"):
        shape = (shape,) if isinstance(shape, int) else tuple(int(n) for n in shape)
        if isinstance(length, (int, float)):
            length = (float(length),) * len(shape)
        length = tuple(float(v) for v in length)
        if not 1 <= len(shape) <= 3 or len(length) != len(shape):
            raise ValueError("shape and length must describe 1 to 3 axes")
        if dtype not in REAL_DTYPE:
            raise ValueError("dtype must be torch.complex64 or torch.complex128")

        self.shape = shape
        self.length = length
        self.ndim = len(shape)
        self.dtype = dtype
        self.real_dtype = REAL_DTYPE[dtype]
        self.device = torch.device(device)
        self.dims = tuple(range(-self.ndim, 0))
        self.dx = tuple(L / n for L, n in zip(length, shape))
        self.dV = math.prod(self.dx)
        self.size = math.prod(shape)

        x64, k64 = [], []
        for axis, (n, L) in enumerate(zip(shape, length)):
            view = [1] * self.ndim
            view[axis] = n
            xi = -L / 2 + torch.arange(n, dtype=torch.float64) * (L / n)
            ki = 2 * math.pi * torch.fft.fftfreq(n, d=L / n, dtype=torch.float64)
            x64.append(xi.reshape(view).to(self.device))
            k64.append(ki.reshape(view).to(self.device))
        self.x64 = tuple(x64)
        self.k64 = tuple(k64)
        self.x = tuple(v.to(self.real_dtype) for v in x64)
        self.k = tuple(v.to(self.real_dtype) for v in k64)
        self.k2 = sum(ki * ki for ki in self.k64).to(self.real_dtype)

    def r2(self):
        """Squared distance from the origin, as a full spatial array."""
        return sum(xi * xi for xi in self.x)

    def fft(self, psi):
        return torch.fft.fftn(psi, dim=self.dims)

    def ifft(self, psi_k):
        return torch.fft.ifftn(psi_k, dim=self.dims)

    def integrate(self, f):
        """Integral over the box, accumulated in double precision.

        Leading (batch, component) axes are kept. The result is float64 (or
        complex128) even for single-precision fields, so diagnostics are not
        limited by float32 summation error.
        """
        acc = torch.complex128 if f.is_complex() else torch.float64
        return f.sum(dim=self.dims, dtype=acc) * self.dV

    def zeros(self, *leading):
        return torch.zeros(*leading, *self.shape, dtype=self.dtype, device=self.device)

    def __repr__(self):
        return (f"Grid(shape={self.shape}, length={self.length}, "
                f"dtype={self.dtype}, device={self.device})")

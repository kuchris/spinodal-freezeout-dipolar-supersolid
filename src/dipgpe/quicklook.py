"""Quick-look plots for debugging. Requires matplotlib; never imported by the core."""

from __future__ import annotations

import numpy as np


def show(grid, psi, *, index=None, title=None, path=None):
    """Plot |psi|^2 and phase of one field.

    ``index`` selects a leading (batch, component) entry; 3D fields show the
    central z slice. Returns the matplotlib figure; saves it when ``path`` is given.
    """
    import matplotlib.pyplot as plt

    field = psi if index is None else psi[index]
    field = field.detach().cpu().numpy()
    if field.ndim != grid.ndim:
        raise ValueError("select a single field with index=")
    if grid.ndim == 3:
        field = field[:, :, field.shape[2] // 2]
    rho, phase = np.abs(field) ** 2, np.angle(field)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if grid.ndim == 1:
        x = grid.x64[0].cpu().numpy()
        axes[0].plot(x, rho)
        axes[1].plot(x, phase)
        axes[0].set_xlabel("x")
        axes[1].set_xlabel("x")
    else:
        Lx, Ly = grid.length[0], grid.length[1]
        extent = (-Ly / 2, Ly / 2, -Lx / 2, Lx / 2)
        im0 = axes[0].imshow(rho, origin="lower", extent=extent, cmap="viridis")
        im1 = axes[1].imshow(phase, origin="lower", extent=extent, cmap="twilight",
                             vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        for ax in axes:
            ax.set_xlabel("y")
            ax.set_ylabel("x")
    axes[0].set_title("|psi|^2")
    axes[1].set_title("arg psi")
    if title:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

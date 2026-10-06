# DipGPE user guide

This guide explains how to set up and solve problems with qs. The README covers
installation and the package layout; docs/RESULTS.md lists validated results
with the commands that reproduce them.

## 1. Units and equation

DipGPE solves

    i ∂ψ/∂t = [ −½∇² + V(x, t) + g|ψ|² + Φ_dd[|ψ|²] + γ|ψ|³ − Ω L_z ] ψ

with ħ = m = 1. For a harmonic trap of angular frequency ω the natural units are
length l = √(ħ/mω), time 1/ω and energy ħω; `dipgpe.units.OscillatorUnits(mass_amu,
omega)` converts: `bohr(a)` (scattering length in l), `per_micron(n)` (linear
density in atoms per l), `microns(x)`. Then g = 4πa_s/l, g_dd = 4πa_dd/l (with
a_dd = μ0 μ_m² m / (12π ħ²)), and `dipgpe.lhy_coefficient(a_s, a_dd)` gives γ.
The norm ∫|ψ|² = N is the atom number.

## 2. Grid, fields and batches

```python
grid = dipgpe.Grid((128, 128, 64), (24.0, 24.0, 12.0), dtype=torch.complex128, device="cuda")
x, y, z = grid.x64          # float64 coordinates, broadcastable
```

The box is periodic, [−L/2, L/2) per axis. Fields keep the spatial axes last;
leading axes are batch axes: a tensor of shape (B, *grid.shape) is B
independent fields evolved or minimized together. complex64 is fast;
complex128 is for validation and precision-sensitive work (Section 9).

Initial fields: `dipgpe.gaussian(grid, center, width, momentum)`,
`dipgpe.normalize(grid, psi, N)`, `dipgpe.translate(grid, psi, shift)`,
`dipgpe.resample(psi, grid_from, grid_to)` (Fourier interpolation between grids),
`dipgpe.fields.imprint_vortices`, `random_vortices`, `random_phase`,
`imprint_vortex_rings`.

## 3. Model terms

```python
model = dipgpe.GPE(grid, dipgpe.Potential(V), dipgpe.Contact(g), dipgpe.Dipolar(grid, g_dd, direction=(0, 0, 1),
               cutoff="cylinder"), dipgpe.LHY(gamma), dipgpe.Rotation(omega))
```

| Term | Meaning |
|---|---|
| `Potential(V)` | external potential: a tensor, or a callable `t -> V(x, t)`; `dipgpe.harmonic(grid, omega)`, `dipgpe.gaussian_obstacle(...)` |
| `Contact(g)` | g\|ψ\|²; g may be a callable `t -> g(t)` (scattering-length ramps) |
| `Dipolar(grid, g_dd, direction, cutoff, radius, height)` | dipole–dipole interaction by FFT convolution (3D only) |
| `LHY(gamma)` | Lee–Huang–Yang term γ\|ψ\|³; γ may be a callable, e.g. `lambda t: dipgpe.lhy_coefficient(a_s(t), a_dd)` |
| `Rotation(omega)` | rotating frame −Ω L_z about z (2D/3D) |
| `Twist(q, axis)` | Bloch phase twist e^{iqx} for superfluid-fraction calculations (ground states only) |

### Dipolar cut-offs

A periodic box would add interactions with image clouds. The kernel is cut off
in real space and transformed analytically (or by quadrature):

- `cutoff="sphere"` (default; Ronen et al.): interaction kept for r < R,
  R = half the smallest box length. The cloud must fit in a box at least twice
  its size in every direction.
- `cutoff="tube"`: for systems periodic along z (a tube): kept for transverse
  distance ρ < R, all z. Dipoles perpendicular to z.
- `cutoff="cylinder"` (dipoles along z): kept for ρ < R and |z| < Z (R = half the
  smaller transverse length, Z = half the box height). For flat clouds the box
  needs to be only about twice the cloud height along z. Condition for no image
  interactions with the defaults: cloud radius < L/4 and full height < L_z/2.
- `cutoff=None`: periodic interaction (only for genuinely periodic systems).

Check the condition on your final state: e.g. the fraction of atoms beyond
ρ = L/4 should be negligible.

## 4. Ground states

```python
psi, info = dipgpe.ground_state(model, psi0, tol=1e-9, preconditioner="combined")
```

Minimizes E[ψ] at fixed N (the norm of `psi0`) by preconditioned nonlinear
conjugate gradient on the sphere ‖ψ‖² = N. It converges to the local minimum
whose basin contains `psi0`: seeds decide which metastable state you get, so
compare several seeds (random vortex positions, different droplet numbers)
when looking for a ground state, and compare converged energies only.

| Option | Meaning |
|---|---|
| `tol` | stop when ‖Hψ − μψ‖/‖ψ‖ < tol (default 1e-9 complex128, 1e-5 complex64, never below 4× the round-off floor ½k²_max ε) |
| `preconditioner` | `"kinetic"` (default) or `"combined"` (kinetic and local potential; usually fewer iterations for trapped clouds and rotating states) |
| `max_angle` | step-angle cap (rad) that keeps the seed's basin (default 0.05) |
| `max_iter`, `strict` | iteration limit; `strict=False` returns the last state instead of raising |
| `callback`, `callback_every` | `callback(it, psi)` returning True stops early |
| `live` | a `dipgpe.live.LiveRun` to watch convergence |
| `reuse` | evaluate the line search from cached operator actions (default True) |

`info` has `iterations`, `mu`, `residual`, `energy`, `converged`, `stopped` and
`capped` (fraction of steps that hit `max_angle`). Imaginary time is also
available (`SplitStep(..., imaginary=True)`) but is much slower.

Superfluid fraction: add `dipgpe.Twist(q, axis)` and use
E(q) − E(0) = f_s N q²/2 (small q; extrapolate in q).

## 5. Time evolution

```python
dt = dipgpe.max_stable_dt(grid)                      # split-step stability limit
stepper = dipgpe.SplitStep(model, dt, dipgpe.STRANG, keep_norm=True)
psi = stepper.evolve(psi, 1000)                  # advances stepper.t
```

`dipgpe.YOSHIDA4` gives fourth order. Above `max_stable_dt` nonlinear runs on a
background density become unstable (a warning is issued). `keep_norm=True`
removes the single-precision cuFFT norm loss (call `evolve` in chunks).
Time-dependent potentials are evaluated at the correct substep times.
`Twist` and other non-local terms except `Rotation` are refused by `SplitStep`.

## 6. Bogoliubov spectra

```python
omega, modes, info = dipgpe.bdg.bogoliubov(model, psi0, q=0.1, axis=-1, n_modes=6, tol=1e-6)
```

Lowest Bogoliubov frequencies of a converged real ground state `psi0` at Bloch
quasi-momentum q along `axis` (for periodic, e.g. supersolid, states use one
unit cell and scan q). Modes are f₋ = u − v; `info["g"]` warm-starts a nearby q
(`x0=`). Rotating or complex states are not supported.

## 7. Observables and analysis

`dipgpe.energy`, `energy_components`, `chemical_potential`, `residual`, `norm`,
`expectation`, `center_of_mass`, `angular_momentum`, `kinetic_energy`;
`dipgpe.vortices.find_vortices` (2D, charges and sub-cell positions),
`find_vortex_points` and `line_components` (3D vortex lines);
`dipgpe.spectra` (incompressible/compressible kinetic energy spectra).

## 8. Run records

Wrap scripts in `with dipgpe.runs.Run("name", vars(args)) as run:` and write with
`run.file`, `run.data`, `run.save_json`, `run.result`. Each run directory holds
the command, parameters, git commit and diff, environment, log, results, data
and a SHA256 manifest; `python -m dipgpe.runs list | find | show | note | backup |
repair` manage them. Results quoted in docs/RESULTS.md cite run ids.

## 9. Precision and performance

- complex128 costs 3.5–7× more than complex64 on consumer GPUs.
- complex64: cuFFT loses ~1e-7 of the norm per FFT pair (`keep_norm`);
  round-off accumulates in neutral modes (soliton position, global phase);
  ground-state residuals cannot go below ~½k²_max ε (the default tolerance
  accounts for this); `ground_state` restores the norm every few iterations.
- Memory: a 512³ complex64 field with the split-step work arrays fits in 16 GB.
- Ground states: prefer `preconditioner="combined"` and the smallest box that
  satisfies the cut-off condition (e.g. the cylindrical cut-off for flat clouds).

## 10. Tests and reproduction

`uv run pytest` runs the physics gates (CUDA if available,
`QS_TEST_DEVICE=cpu` forces the CPU). `examples/sample_run.py` is a short
end-to-end check with expected output. Every benchmark in `benchmarks/` writes
a run record; docs/RESULTS.md gives the command and run id for each result.

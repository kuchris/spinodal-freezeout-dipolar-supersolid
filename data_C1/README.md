# Reduced data for "Spinodal-controlled freeze-out and nucleation at the
first-order superfluid--supersolid transition of a dipolar gas" (W. K. Wong)

Per-realization observables of the truncated-Wigner simulations (no
wavefunctions). Every run directory holds `params.json` (the full parameter set,
command, git commit and environment of the run) and one `tau_<t>ms.npz` per
ramp time. `index.csv` lists all files with density, protocol, noise and the
role of the run in the paper.

Arrays in `tau_<t>ms.npz` (R = realizations, K = records, every `record_ms`):

| name | shape | meaning |
|---|---|---|
| times_ms | (K,) | time since the start of the run (ms); the ramp starts at `equil_ms` |
| fs | (K, R) | Leggett superfluid fraction of the line density, L^2 / (N int dz / n(z)) |
| contrast | (K, R) | line-density contrast (max - min) / (max + min) |
| peaks | (K, R) | local maxima of n(z) above 1.2 x mean (droplet count) |
| deep_peaks | (K, R) | local maxima above 1.5 x mean (later runs only) |
| fs_avg_1ms | (K, R) | f_s of the line density averaged over 1 ms (later runs only) |
| g2 | (K2, Nz/2) | sample-averaged density correlation g2(dz), every `store_every` records |
| g2_times_ms | (K2,) | times of g2 and line |
| line | (K2, S, Nz) | line density n(z) (atoms per oscillator length l = 0.641 um) of the first S realizations |
| dz_um | () | grid spacing along the tube in um |

The scattering length follows a_s(t) = a_i + (a_f - a_i) min(1, max(0, (t - equil_ms)/tau)),
with a_i, a_f, equil_ms in params.json (a_s in Bohr radii). Spinodals and
transition points: a_rot* = 89.845, 92.314, 84.506 a0 and a* = 90.256, 92.314,
85.28 a0 at 6250, 2500, 700 per um; end of the crystal branch 90.39 a0 at 6250 per um.

`static/` holds the ground-state energies (tube_first_order), the Bogoliubov
roton and crystal soft-mode results and the ground states of Fig. 1(d), each with
its run metadata.

Analysis and figures: the scripts benchmarks/tube_ramp_errors.py,
tube_ramp_figures.py, tube_ramp_nucleation.py, tube_ramp_cutoff.py and
tube_ramp_dt_check.py of the dipgpe code read the original run records; the
arrays here are the same observables in a portable format.

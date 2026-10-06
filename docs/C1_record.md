# Paper C1: results and issues log

Extracted from the development log of DipGPE (newest first). Run ids refer to
`runs/<id>_*` (metadata, logs and results in this repository; the per-realization
observables are in the data archive, see README). Some run records name the
former package name `qs` and the former script `tube_bdg.py roton` (now
`tube_roton.py`); the code is otherwise identical.

## 2026-10-06 — Time-step convergence (`benchmarks/tube_ramp_dt_check.py`, run `20261006-144331_tube_ramp_dt_check`)

The grid comparison (dx 0.25 → 0.1875 l) also changes the stability-limited
time step, so the time step was tested on its own (point raised by the
independent Codex reanalysis, ISSUES C1-5): 6250/µm forward ramps, τ_Q = 20 and
50 ms, 16 realizations, seed 0 (identical noise to the production run
`20261005-132617`), `--dt-safety 0.45` (dt 10 → 5 µs), run `20261006-142433`.

| τ_Q | f_s < 0.98 | f_s < 0.9 | f_s < 0.7 | largest per-realization change |
|---|---|---|---|---|
| 20 ms | 8.925 → 8.925 ms | 10.225 → 10.225 | 10.925 → 10.925 | 0.1 ms |
| 50 ms | 12.462 → 12.462 ms | 14.162 → 14.162 | 15.162 → 15.162 | 0.1 ms |

The time-discretization error of the onsets is below the 0.1 ms record
interval; no temporal term is needed in the error budget.

## 2026-10-06 — Fit window: onsets after the end of the ramp (ISSUES C1-5)

An independent reanalysis (Codex) found that the 6250/µm,
τ_Q = 10 ms onset (median 7.5 ms after the spinodal) and the 700/µm, τ_Q = 10, 20 ms
onsets happen after the ramp has stopped. `tube_ramp_errors.py` now fits only the
τ_Q whose median onset precedes the ramp end and tests thresholds on a common
window (run `20261006-142523_tube_ramp_errors`, identical to the Codex numbers):

| data set | ζ | fit window | all points | bootstrap | threshold ½-range | reference | window | total |
|---|---|---|---|---|---|---|---|---|
| 6250 forward | 0.363 | 20–200 ms | 0.342 | 0.0074 | 0.0100 | 0.0080 | 0.0010 | 0.0211 |
| 2500 forward | 0.364 | 10–200 ms | 0.364 | 0.0064 | 0.0044 | 0.0047 | 0.0061 | 0.0186 |
| 700 forward (descriptive) | 0.221 | 50–400 ms | 0.262 | 0.0260 | 0.0131 | 0.0209 | 0.0000 | 0.0389 |
| 6250 reverse | 0.351 | 10–200 ms | 0.351 | 0.0016 | 0.0096 | 0.0168 | 0.0000 | 0.0245 |

The forward 10 ms onset in the hysteresis plot is at the held 88.00 a0 (not the
extrapolated 86.84 a0); hysteresis width 3.545 a0 (10 ms) to 1.12 a0 (200 ms).

## 2026-10-06 — Statistics extended (four additional ensembles; error run `20261006-025952_tube_ramp_errors`)

New runs (16 realizations unless stated, complex64, same parameters as the
production runs, new seeds): (i) `20261006-012712` 2500/µm forward τ_Q = 200 ms;
(ii) `20261006-014539` 6250/µm reverse τ_Q = 200 ms; (iii) `20261006-020234` 5 nK
thermal ramps 92 → 88 a0, τ_Q = 50, 100 ms with deep counts (seed 500); (iv)
`20261006-021817` 5 nK, 92 → 89.6 a0 in 120 ms (τ_Q-equivalent 200 ms, seed
600); (v) `20261006-022814` 700/µm τ_Q = 300, 400 ms, 8 more each (seed 7).
Wall time 01:27–02:59 (1.5 h). `tube_ramp_figures.py` now pools realizations
of the same τ_Q over runs.

| data set | ζ (f_s 0.98) | bootstrap | thresholds 0.99/0.98/0.95/0.90 | reference ±0.01 a0 | total | τ_Q |
|---|---|---|---|---|---|---|
| 6250 forward | 0.342 | 0.005 | 0.347/0.342/0.334/0.329 | 0.007 | 0.019 | 10–200 |
| 2500 forward | 0.364 | 0.006 | 0.360/0.364/0.367/0.370 | 0.005 | 0.018 | 10–200 (was 10–100: 0.369) |
| 700 forward | 0.262 | 0.013 | 0.258/0.262/0.263/0.262 | 0.013 | 0.024 | 10–400 (was 0.253) |
| 6250 reverse | 0.351 | 0.002 | 0.346/0.351/0.364/0.365 | 0.017 | 0.025 | 10–200 (was 10–100: 0.360) |

- Medians: 2500/µm τ_Q = 200 ms t̂ = 14.55 ms (12.95–15.65); reverse
  τ_Q = 200 ms melting delay 9.40 ms (9.2–9.5), transition at 90.55 a0;
  700/µm τ_Q = 300, 400 ms (16 each): 18.85 ms (8.35–21.85; one early
  realization) and 17.20 ms, final droplets 28.1 and 24.4.
- Ramp test pooled (deep counts, droplets before the spinodal, 5 nK, from
  92 a0): 0/16 (τ_Q 50 ms, hold prediction 0.07), 1/16 (100 ms, 0.13), 2/32
  (200 ms, 0.25; binomial P(≤ 2 | 32, 0.25) = 0.007). The hold rates
  over-predict slow ramps from 92 a0 significantly.
- f_s < 0.7 after the spinodal, 5 nK pooled (24 realizations) vs quantum (16):
  13.56 vs 15.16 ms (τ_Q 50), 16.03 vs 19.03 ms (τ_Q 100); earliest 5 nK
  realization 0.53 ms after the spinodal (none before it).

## 2026-10-06 — Soft mode at the end of the crystal branch (`benchmarks/tube_crystal_bdg.py`)

Crystal cell of the 3D ramps (6250/µm, cell 2.76 µm, 96 × 96 × 17, dx = 0.25 l,
L_perp = 24 l, complex128), continued upwards in a_s; BdG at q = 0 with the
translation mode deflated (ISSUES BD-6), tol 1e-4. Runs `20261006-002643`
(90.0, 90.2; stopped at 90.3 by BD-6 before the fix), `-011016` (90.3),
`-005533` (90.33–90.40); zone checks `-010515`.

| a_s (a0) | 90.0 | 90.2 | 90.3 | 90.33 | 90.36 | 90.38 | 90.40 |
|---|---|---|---|---|---|---|---|
| line contrast | 0.785 | 0.734 | 0.692 | 0.674 | 0.648 | 0.615 | lost (0) |
| soft mode ω (q = 0, ħω⊥) | 0.674 | 0.535 | 0.421 | 0.369 | 0.292 | 0.175 | — |

- The soft mode is the q = 0 amplitude mode of the crystal (same period as the
  cell); the next q = 0 modes are the transverse Kohn modes at 1.0000 (check).
- ω⁴ is linear in a_s near the end (last 4 points: residual 1.8e-6 vs 2.6e-4 for
  ω² linear), zero at a_end = 90.381–90.382 a0; a free fit ω ∝ (a_end − a)^p
  gives p = 0.29–0.31, a_end = 90.384. Saddle-node (fold) of the crystal branch:
  ω² ∝ (a_end − a)^(1/2), p = 1/4. a_end agrees with the static branch end 90.39
  used as the melting reference (shift 0.006–0.009 a0).
- No finite-q instability first: at q = π/L and π/(2L) the lowest bands are the
  two gapless (phonon) branches, 0.419/0.613 (90.3) and 0.451/0.555 (90.38) at
  the zone edge; they do not soften towards the end.
- Consequence for the melting exponent (reverse ramps, ζ = 0.36 ± 0.02): a
  single-mode description of the slow passage through a fold gives t̂ ∝ τ_Q^(1/5)
  for conservative dynamics (ẍ = −rt − x², equivalently KZ with ω ∝ ε^(1/4)) and
  τ_Q^(1/3) for overdamped dynamics (ẋ = −rt − x², the mean-field dynamical
  hysteresis law). The measured exponent is far from 1/5 and close to 1/3: the
  melting does not follow the undamped soft mode; a plausible reason is that the
  soft mode decays into the phonon branches (effective damping) once the
  crystal is driven past the fold. Not tested.

## 2026-10-06 — Noise-cutoff sensitivity (`benchmarks/tube_ramp_cutoff.py`, run `20261006-000847_tube_ramp_cutoff`)

`tube_ramp.py --noise-cutoff c` scales the quantum k_max = 1/ξ or the thermal
energy cutoff 2kT. 6250/µm, 3D, 16 realizations, complex64.

Quantum noise, forward ramps 92 → 88 a0, seed 0 (runs `20261005-230857` c = 0.7,
`20261005-232442` c = 1.3; c = 1 from the production runs), median t̂ (f_s <
0.98 after a_rot* = 89.845):

| c | t̂ (τ_Q 20 ms) | range | t̂ (τ_Q 100 ms) | range | two-point ζ |
|---|---|---|---|---|---|
| 0.7 | 9.03 | 8.43–9.93 | 15.52 | 14.62–17.12 | 0.337 |
| 1.0 | 8.93 | 8.33–9.53 | 15.92 | 14.72–16.92 | 0.360 |
| 1.3 | 9.13 | 8.63–9.63 | 15.82 | 15.12–16.82 | 0.342 |

Thermal noise, holds at 90.05 a0, 5 nK, from 92 a0 in 20 ms, 150 ms, seed 200
(runs `20261005-234029` c = 0.7, `20261005-235454` c = 1.3):

| c | nucleated | Γ (µm⁻¹ s⁻¹) | roton power at the hold start | added atoms |
|---|---|---|---|---|
| 0.7 | 13/16 | 0.15 ± 0.04 | 0.029 | 0.39% |
| 1.0 | 12/16 | 0.11 ± 0.03 | 0.054 | 0.40% |
| 1.3 | 13/16 | 0.14 ± 0.04 | 0.055 | 0.41% |

- t̂ changes by ≤ 0.4 ms (2.5%), less than the realization spread; ζ varies
  by ≤ 0.02 without a trend (within the quoted ±0.019). Expected: the growth
  time depends only logarithmically on the seed amplitude.
- Γ is unchanged within errors. At c = 0.7 the cutoff (1.4 kT = 0.97 ħω) lies
  below the free-particle roton energy (k_rot²/2 = 1.14 ħω ≈ 1.65 kT), so the
  roton is not seeded directly and its power at the hold start halves, yet Γ is
  the same: the nucleation seeds are generated dynamically from the other
  thermal modes during the 20 ms ramp and the hold.
- Scope: two τ_Q values and one (a_s, T) point; rules out cutoff effects larger
  than ~0.02 in ζ and ~40% in Γ.

## 2026-10-05 — Nucleation rate inside the metastable window (`benchmarks/tube_ramp_nucleation.py`)

Holds at 6250/µm (3D, 32 cells = 88 µm, complex64): ramp 92 → a_f in 20 ms,
hold 150 ms, thermal truncated-Wigner noise, deep-droplet count (> 1.5 × mean
line density) recorded every 0.2 ms. New runs with 16 realizations, seed 200:
`20261005-205640` (89.95, 5 nK), `-211050` (90.05, 5 nK), `-212444` (90.15,
5 nK), `-213927` (90.05, 2.5 nK), `-215509` (90.05, 10 nK); pooled with the
earlier 8-realization deep-count runs `-191921` (2.5 nK) and `-192559` (10 nK).
Waiting time t₁ from the crossing of a* to the first deep droplet, right-censored
at the end of the run; MLE ΓL = events / exposure, error Γ/√events. Delay =
earliest t₁ (MLE shift of a delayed exponential). Analysis run
`20261005-230215_tube_ramp_nucleation` (fig_survival.png, nucleation.json).

| a_s (a0) | T (nK) | nucleated | Γ (µm⁻¹ s⁻¹) | delay (ms) | Γ, delayed exp. |
|---|---|---|---|---|---|
| 89.95 | 5 | 16/16 | 0.32 ± 0.08 | 7.6 | 0.41 ± 0.10 |
| 90.05 | 5 | 12/16 | 0.11 ± 0.03 | 7.9 | 0.12 ± 0.04 |
| 90.15 | 5 | 6/16 | 0.039 ± 0.016 | 9.3 | 0.043 ± 0.017 |
| 90.05 | 2.5 | 10/24 | 0.039 ± 0.012 | 43.5 | 0.060 ± 0.019 |
| 90.05 | 10 | 24/24 | 0.52 ± 0.11 | 2.3 | 0.58 ± 0.12 |

- Survival curves are consistent with exponentials after a short delay (the
  growth time to a detectable droplet), so Γ is an effective rate of detectable
  nuclei.
- Γ grows ×8 across 0.2 a0 of the window (near a* → near the spinodal) and ×13
  from 2.5 to 10 nK (closer to Γ ∝ T^~2 than to a constant activation energy
  over this narrow range; no Arrhenius claim).
- The older loose-count (> 1.2 × mean) estimates at 5 nK are 20–50% higher
  (0.45, 0.12, 0.059, 0.043 at 89.95, 90.05, 90.15, 90.25): loose peaks include
  thermal fluctuations right after the crossing of a*.
- Prediction for 5 nK ramps 92 → 88 a0 (rate 4/τ_Q a0/ms), Γ linear in a_s
  between the holds, Γ(a*) = 0, flat from 89.95 to the spinodal: expected
  nuclei m = L∫Γ da / r, P = 1 − e^(−m) = 0.03, 0.07, 0.13, 0.25, 0.43, 0.68
  for τ_Q = 20, 50, 100, 200, 400, 800 ms; crossover P = 1/2 at τ_Q^× ≈ 490 ms.
- Earlier 5 nK thermal ramps (run `20261005-175324`, loose count, 8
  realizations): droplets before the spinodal in 0/8 (τ_Q = 50 ms) and 2/8
  (100 ms); predicted 0.07 and 0.13. Of the two at 100 ms, one is a rare
  realization with a large density modulation already at 91.4 a0 (contrast 0.4,
  f_s 0.98) that crystallizes as soon as it crosses a*; the other is a
  loose-count detection right at a*.

Ramp tests and preparation dependence (16 realizations, deep count, 5 nK; analysis
run `20261005-230215_tube_ramp_nucleation`). Roton power = sample-averaged
line-density structure factor summed over the roton band (Fourier indices 28–38).

| run | protocol | τ_Q-equivalent | droplet before the spinodal | predicted from holds | roton power at a* |
|---|---|---|---|---|---|
| `20261005-223121` | 92 → 89.6 a0 in 120 ms | 200 ms | 1/16 | 0.25 | 0.039 |
| `20261005-221252` | 90.5 → 89.6 a0 in 45 ms | 200 ms | 11/16 | 0.25 | 0.105 |
| `20261005-221252` | 90.5 → 89.6 a0 in 90 ms | 400 ms | 10/16 | 0.43 | 0.227 |

Hold control from 90.5 a0 (run `20261005-224452`, 90.5 → 90.05 in 20 ms, hold 150
ms): Γ = 0.079 ± 0.025 µm⁻¹ s⁻¹ (10/16), roton power at the hold start 0.111,
vs 0.108 ± 0.031 and 0.054 from 92 a0.

- No false positives: no deep droplet before t(a*) in any realization of the
  90.5 a0 ramps; by the spinodal crossing the early realizations carry 8–33
  deep droplets.
- At the same ramp rate through the window, starting at 90.5 a0 instead of
  92 a0 raises early nucleation from 1/16 to 11/16. The roton band carries
  2.7× more power at a* (0.105 vs 0.039) when the noise is injected close to the window: the
  free-particle-basis thermal noise projects onto the soft Bogoliubov mode with
  a weight (ε² + ω²)/(2εω) that grows as ω_rot falls (ω_rot ≈ 0.62 at 92 a0 and
  0.34 at 90.5 a0 from the BdG fit ω_rot² ∝ a_s − a_rot*), and without a bath the
  roton population then follows adiabatically instead of growing to kT/ω.
- The ramp from 92 a0 (the protocol of all production ramps) nucleates less
  than the hold rates predict (1/16 vs 0.25; binomial probability of ≤ 1 event
  is 6%): the holds enter the window by a fast 20 ms ramp, which excites the
  roton more than a slow ramp. Hold rates are therefore an upper estimate for
  slow ramps from 92 a0, and at 5 nK the crossover in that protocol lies above
  τ_Q ≈ 0.5 s.
- 2.5 / 5 / 10 nK holds: roton power at the hold start 0.002 / 0.05 / 0.5–1.3.
- Conclusion: within truncated Wigner without a bath, Γ and τ_Q^× depend on
  the preparation (ISSUES C1-4). The qualitative picture is robust — fast
  ramps and low temperature follow the spinodal route, finite temperature
  allows nucleation in the window — but quantitative rates need a thermalizing
  c-field method (SPGPE) or Bogoliubov-basis thermal sampling.

## 2026-10-05 — First 3D ramps, first-order (6250/µm) vs continuous (2500/µm) (pilot)

Full 3D eGPE, 96² × N_z grid (dx = 0.25 l, L_perp = 24 l), 32 cells, complex64,
quantum noise, 4 realizations, records every 0.2 ms. Cost: 0.8 ms per step per
realization for 16 cells (cuFFT warm-up excluded). Freeze-out t̂ (ms):

| τ_Q (ms) | 6250/µm, 92 → 88 a0, t_c at a_rot* = 89.83 (run `20261005-131531`) | 2500/µm, 96 → 88 a0, t_c at 92.32 (run `20261005-132026`) |
|---|---|---|
| 20 | 8.55 | 6.40 |
| 50 | 11.68 | 9.20 |
| 100 | 14.95 | 11.40 |

At 6250/µm, referenced to the spinodal a_rot*, the local slopes (0.34, 0.36) are
KZ-like; referenced to a* = 90.256 they would be 0.5–0.57. So with quantum
noise only the uniform state survives through the metastable window and the
crystal forms only after the roton softens (no early nucleation): a "spinodal"
Kibble–Zurek picture. Pilot only (4 realizations).

Production at 6250/µm (16 realizations, records every 0.1 ms; runs
`20261005-132617`, τ_Q = 10, 20, 50 ms, and `20261005-133951`, 100, 200 ms):
t̂ = 7.28, 8.75, 11.98, 15.25, 19.50 ms after the spinodal crossing (a_rot* =
89.83 a0): ζ = 0.332 ± 0.009 (mean-field KZ: 1/3). Referred to a* = 90.256 a0:
8.34, 10.88, 17.30, 25.90, 40.80 ms, ζ = 0.531 ± 0.028 (curved in log-log). So
with quantum noise the freeze-out follows Kibble–Zurek scaling measured from the
spinodal, as for a continuous transition. The g2 envelope X is too noisy with
16 3D realizations (exponent 0.40 ± 0.21): needs more realizations or another
estimator. Cost: 13 + 27 min.

Continuous control in 3D, 2500/µm, 96 → 88 a0, t_c at 92.32 a0, 16
realizations (run `20261005-140819`): t̂ = 4.9, 6.4, 8.9, 11.3 ms for τ_Q = 10,
20, 50, 100 ms, ζ = 0.362 ± 0.006 (quasi-1D anchor 0.355(4); Kirkby et al.'s 3D
check 0.33(1)); g2 envelope exponent 0.352 ± 0.049. Final droplet number
33.1–34.1 for a 32-cell ground state (the ramp selects a shorter period, fitted
2.48–2.76 µm vs the ground-state cell 2.83 µm). The first-order 6250/µm value
(0.332(9), from the spinodal) is 0.03 lower — within the a_c systematics (ζ moves
by ~0.04 per 0.03 a0 of t_c reference at these τ_Q) until a_rot* on the dynamical
grid and its uncertainty are pinned down. (The ramp spans differ: 8 a0 here,
4 a0 at 6250, so absolute t̂ at equal τ_Q are not directly comparable.)

a_rot* on the dynamical grid (dx = 0.25 l, L_perp = 24 l; BdG of the uniform
tube, ω_rot² linear in a_s; `tube_roton.py --density`): 89.845 a0 at
6250/µm (static value used above 89.83; roton wavelength 4.16 l = 2.67 µm) and
92.314 a0 at 2500/µm (92.32). With these: ζ = 0.343 ± 0.010 (6250, first order,
from the spinodal) and 0.360 ± 0.006 (2500, continuous): consistent within
1.5σ. So far: Kibble–Zurek freeze-out measured from the spinodal, with the same
exponent as at the continuous transition.

700/µm (first order, direct BEC → droplet crystal; a_rot* = 84.506 a0 on the
dynamical grid, roton wavelength 5.0 l = 3.2 µm; a* = 85.28), 87 → 83 a0, 32
cells of 3.2 µm, 16 realizations (runs `20261005-145757`, `-151305`, `-152429`):
t̂ = 6.97, 7.93, 11.43, 13.45, 15.60 ms from the spinodal for τ_Q = 10, 20, 50,
100, 200 ms: ζ = 0.283 ± 0.020 (local slopes 0.19, 0.40, 0.24, 0.21: noisy);
from a*: 8.9 … 54.3 ms, ζ = 0.610 ± 0.030. Final state an insulating droplet
crystal (<f_s> 0.09–0.16, contrast ≈ 1, 31–33 droplets for 32 cells). The
spinodal exponent is ~3σ below 6250 and 2500: either a real difference at the
direct BEC–droplet transition or noise from 16 realizations — needs more
realizations before any claim.

Reverse ramps (pilot, 6250/µm, crystal ground state at 88.5 a0 tiled over 32
cells, 88.5 → 92 a0, 4 realizations, run `20261005-154515`): <f_s> exceeds 0.98
at a_s = 91.14, 90.81, 90.66 a0 for τ_Q = 20, 50, 100 ms, approaching the static
end of the crystal branch (90.375–90.40) from above, i.e. the crystal survives
past a* = 90.256 to its own spinodal. Melting delay after 90.39: 4.3, 6.0, 7.6 ms
(local slopes 0.36, 0.34, KZ-like). Together with the forward ramps (uniform
state kept to a_rot* = 89.845) this is a dynamical hysteresis loop: static
window 89.85–90.39 a0 plus delays that shrink as τ_Q^~0.35. <f_s> already
passes 0.5 near 89.65 a0 (the supersolid branch's f_s grows with a_s), so only
the final rise marks melting.

Reverse-ramp production at 6250/µm (16 realizations, run `20261005-155046`):
melting delay after the branch end 90.39 a0: 3.3, 4.3, 5.9, 7.6 ms for τ_Q = 10,
20, 50, 100 ms, ζ = 0.360 ± 0.005; referred to a* = 90.256 a0 the data are
curved (ζ ≈ 0.49). So both directions scale from their own spinodal (roton
softening of the uniform state going down, end of the crystal branch going
up) with the same exponent as the continuous transition (0.343(10) and
0.360(6)). The dynamical hysteresis width therefore exceeds the static window
(89.85–90.39 a0) by rate × (t̂_down + t̂_up) ∝ τ_Q^(ζ − 1) ≈ τ_Q^(−0.65).
(The g2 "correlation length" fit has no meaning for melting and is not used.)

Reverse ramps at 700/µm (droplet crystal at 83 a0 in 3.2 µm cells, contrast
0.972, <f_s> = 0.16; 83 → 87 a0, 20 ms hold, 16 realizations, run
`20261005-161430`, τ_Q = 10, 20, 50 ms): the crystal does not melt cleanly. <f_s>
passes 0.5 near 84.5–85.3 a0 but ends at 0.82, 0.94, 0.79, and 32, 21.4, 24.9
droplets of 32 remain (partial merging) although a_s = 87 is well above a* =
85.28 and the static branch end 85.51 (best cell). At low density the
self-bound droplets survive far beyond the static branch end on these time
scales: the hysteresis is much larger than at 6250/µm, and "<f_s> > 0.98" is
not a usable melting criterion here. By droplet number (peaks > 1.2 × mean),
the count falls below 90% / 75% / 50% of 32 at a_s = 86.04 / 86.12 / 86.20
(τ_Q = 10 ms), 85.36 / 85.54 / never (20 ms) and 84.66 / 84.95 / 85.70 (50 ms):
for slow ramps the droplets start merging already below the uniform spinodal
(84.506) and a* (85.28). At low density raising a_s first coarsens the droplet
crystal (the equilibrium spacing grows towards a*), rather than melting it.
At 10 ms the count returns to 32 after melting, probably density oscillations
counted by the loose peak threshold (this run predates the "deep" count).
A further reverse ramp 83 → 89 a0 (τ_Q = 50 ms, 60 ms hold, 8 realizations,
run `20261005-195954_tube_ramp_3d`): deep droplets 32 → 15.2 (85.4 a0) → 8.2 (87.8 a0) and then ~7–8
throughout the hold at 89 a0 (3.7 a0 above a*), with the 1 ms averaged f_s
0.92–0.93. About a quarter of the droplets are long-lived self-bound objects
far above the transition; the rest merge first and then dissolve.

Convergence of the forward freeze-out at 6250/µm, τ_Q = 50 ms, t_c from a_rot* =
89.845 a0: 32 cells 12.16 ms (16 realizations; two halves 12.06 and 12.16);
64 cells 12.26 ms (16 realizations, run `20261005-162920`): tube length
converged to < 1%. dx 0.1875 l (128² × 736): a_rot* = 89.8450 (89.8449 at
0.25) and t̂ = 11.86 ms (8 realizations, run `20261005-165050`): −2.5%, larger
than the half-to-half spread, so a small discretization shift; it does not
change ζ if it is a constant fraction (to check at a second τ_Q).
Transverse box 24 → 32 l (128² × 552): a_rot* = 89.8416 a0 (89.845) and median
t̂ = 12.22 ms vs 12.46 ms (8 vs 16 realizations, run `20261005-190818_tube_ramp_3d`): −2%, the size of
the dx shift and of the realization spread. Precision: the same 4 noise seeds
(16 cells, τ_Q = 50 ms) in complex64 (run `20261005-191441_tube_ramp_3d`) and complex128 (run `20261005-191531_tube_ramp_3d`)
give identical per-realization t̂ (12.66, 11.96, 10.96, 11.86 ms); f_s differs
by ≤ 3e-7 before t_c and by ≤ 0.012 after (chaotic divergence), final <f_s>
0.35468 vs 0.35474. complex64 is adequate; complex128 costs 3.9×.
dx at a second τ_Q (median t̂, 8 realizations at dx 0.1875 vs 16 at 0.25):
τ_Q = 50 ms 12.26 vs 12.46 ms (−1.6%), τ_Q = 100 ms 15.83 vs 15.93 ms (−0.6%,
run `20261005-193642_tube_ramp_3d`): ζ moves by ≈ +0.015, within its error.

700/µm with 16 more realizations from other noise seeds (run
`20261005-170414`): t̂ = 7.165, 8.13, 11.625 ms for τ_Q = 10, 20, 50 ms, i.e.
the earlier values + 0.20 ms each, with the same local slopes (0.18 for
10 → 20 ms). With 0.06% added atoms the freeze-out is set by the deterministic
growth of the unstable mode; the realizations change it little. So the curved
t̂(τ_Q) at 700/µm is not statistical noise: in 10–200 ms it is not a single
power law. (`tube_ramp_analysis.py --pool` pools realizations of equal τ_Q
across runs.)

Longer ramps at 700/µm (8 realizations, run `20261005-171904`) and per-realization
analysis (median over realizations of each one's f_s < 0.98 crossing,
`--estimator median`):

| τ_Q (ms) | 10 | 20 | 50 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|
| median t̂ (ms) | 7.27 | 8.13 | 11.62 | 13.45 | 16.10 | 17.95 | 16.80 |
| spread min–max (ms) | 6.1–8.1 | 7.1–9.0 | 10.1–13.4 | 12.4–15.0 | 13.8–19.5 | 8.4–21.4 | 14.9–19.6 |
| a_s at the first droplet (a0) | 83.00 | 83.66 | 83.98 | 84.24 | 84.38 | 84.44 | 84.46 |
| final droplets (32 cells) | 31.1 | 32.6 | 31.6 | 31.9 | 30.1 | 28.1 | 24.0 |

The mean-based t̂ at 300 ms (13.65) was pulled down by one realization that
formed droplets early (8.35 ms); medians increase up to 300 ms and then level
off. ζ (median) = 0.253 ± 0.019 at 700/µm against 0.342 ± 0.012 (6250) and
0.369 ± 0.005 (2500), where mean and median agree. For slow ramps the first
droplets appear at 84.38–84.46 a0, just below the spinodal 84.506: no early
nucleation, as at 6250, but the global f_s drops ~10 ms later because the
isolated droplets form locally and spread. The final droplet number falls from
31–33 to 24 as τ_Q grows: slower ramps select larger spacings (the equilibrium
spacing diverges near a* at this density), a KZ observable specific to the
low-density first-order transition.

Thermal-noise control at 6250/µm (T = 5 nK: modes below 2 kT with
n_BE + 1/2, transverse oscillator modes included, adds 0.40% atoms; 8
realizations; run `20261005-175324`). Thermal roton fluctuations already lower
the Leggett bound near the spinodal (and push "droplet" peaks above 1.2 × mean
at 90.97 a0, above a*), so the 0.98 threshold is contaminated. Per-realization
crossings, median (min) relative to the spinodal crossing:

| threshold | τ_Q = 50 ms quantum | 50 ms, 5 nK | 100 ms quantum | 100 ms, 5 nK |
|---|---|---|---|---|
| f_s < 0.98 | 12.46 (10.86) | 10.96 (4.06) | 15.93 (14.72) | 10.72 (−29.08) |
| f_s < 0.9 | 14.16 (12.66) | 12.76 (5.86) | 17.82 (16.82) | 13.12 (−8.17) |
| f_s < 0.7 | 15.16 (13.76) | 13.76 (12.26) | 19.03 (18.03) | 14.93 (2.43) |

With thermal noise the transition comes earlier, more so for the slower ramp
(median −9% at 50 ms, −22% at 100 ms with f_s < 0.7). With the strictest
threshold (0.7) no realization transforms before the spinodal; the one at
−8 ms (f_s < 0.9, a_s ≈ 90.17 a0, inside the window) may be a large thermal
roton fluctuation. Explanations not yet separated: (i) larger seeds (thermal
≫ quantum) shorten the growth after the spinodal; (ii) the softening roton's
thermal occupation (∝ kT/ε_rot) grows before the spinodal, more for slow
ramps; (iii) thermally activated nucleation inside the window. Preliminary
(8 realizations); needs an order parameter that thermal roton fluctuations do
not mimic and more realizations before any claim about nucleation.

Decisive test: ramp 92 → 90.05 a0 in 20 ms and hold at 90.05 a0 for 150 ms,
inside the metastable window (above the spinodal 89.845, below a* = 90.256;
the uniform state is linearly stable there, BdG ω_rot ≈ 0.19 ħω), 8
realizations each:

| t (ms) | quantum noise (run `20261005-181025`) | 5 nK thermal noise (run `20261005-180330`) |
|---|---|---|
| 30 | <f_s> 1.000, contrast 0.04, 0 droplets | 0.997, 0.11, 0 |
| 60 | 1.000, 0.04, 0 | 0.929, 0.37, 7.9 (max 26) |
| 100 | 1.000, 0.04, 0 | 0.833, 0.48, 14.1 (max 34) |
| 180 | 1.000, 0.04, 0 | 0.739, 0.67, 21.8 (max 32) |

With quantum noise the uniform state survives the whole hold; with thermal
noise the crystal appears region by region and grows during the hold. Since the
uniform state is linearly stable at 90.05 a0, larger seeds alone cannot do
this: it is thermally activated nucleation and growth inside the window
(explanation (iii) above), in the classical-field (truncated-Wigner) sense at
5 nK. So: at T = 0 the first-order transition happens at the spinodal with KZ
scaling; at finite T nucleation inside the window competes, more for slow
ramps. Limits: 8 realizations, temperature not controlled during the
evolution (classical-field dynamics).

Holds across the window, 5 nK thermal noise, 8 realizations each (runs
`20261005-181821_tube_ramp_3d` 89.95, `20261005-182509_tube_ramp_3d` 90.15, `20261005-183155_tube_ramp_3d` 90.25 a0), mean droplet number:

| hold a_s (a0) | 60 ms | 100 ms | 180 ms | final <f_s> |
|---|---|---|---|---|
| 89.95 (0.10 above the spinodal) | 12.9 | 23.4 | 32.0 | 0.60 |
| 90.05 | 7.9 | 14.1 | 21.8 | 0.74 |
| 90.15 | 6.0 | 10.9 | 12.5 | 0.87 |
| 90.25 (≈ a* = 90.256) | 4.1 | 8.0 | 8.0 | 0.93 |
| 90.05, quantum noise | 0 | 0 | 0 | 1.00 |

Nucleation and growth slow down monotonically from the spinodal towards a*;
at a* the nucleated droplets stop growing (no free-energy gain), the count
saturates near 8. The first ≥ 3 droplets appear 6–9 ms after the hold starts
(35.6–38.6 ms; the ramp ends at 30 ms) at every a_s. Same limits as above (classical-field noise, 8 realizations).

Temperature dependence at the 90.05 a0 hold (8 realizations; 2.5 nK run
`20261005-191921_tube_ramp_3d`, 10 nK run `20261005-192559_tube_ramp_3d`; these record two order parameters that thermal roton
fluctuations do not mimic: droplets denser than 1.5 × the mean ("deep") and the
Leggett bound of the 1 ms running average of n(z)). Droplets (deep in brackets):

| T | added atoms | 40 ms | 60 ms | 100 ms | 180 ms | f_s(1 ms) at 180 ms |
|---|---|---|---|---|---|---|
| 0 (quantum) | 0.01% | 0 | 0 | 0 | 0 | 1.000 (instantaneous) |
| 2.5 nK | 0.19% | 0 (0) | 0 (0) | 5.4 (2.6) | 18.2 (16.2) | 0.789 |
| 5 nK | 0.40% | 1.9 | 7.9 | 14.1 | 21.8 | 0.739 (instantaneous) |
| 10 nK | 0.84% | 11.8 (7.0) | 23.6 (15.8) | 29.0 (26.0) | 31.2 (30.0) | 0.624 |

Nucleation starts earlier and proceeds faster as T grows, and the droplets are
deep (not fluctuations): thermally activated nucleation inside the metastable
window, absent at T = 0 on these time scales.

## 2026-10-05 — Kibble–Zurek anchor at n = 2500/µm (`benchmarks/tube_ramp.py`, `tube_ramp_analysis.py`)

Reproduction of Kirkby et al. (PRR 7, 2025) in their quasi-1D model (closed
kernel, l = 1.08 µm, η = 4.25, L = 344 µm = 128 cells, a: 96 → 88 a0, 10 ms hold
before the ramp), quantum noise only (half a quantum per plane wave with
|k| < 1/ξ; adds 0.02% atoms), 64 realizations, f_s recorded every 0.1 ms,
τ_Q = 20, 50, 100, 200, 400, 770 ms (run `20261005-125247`; a first pass with 32
realizations and 0.5 ms records, `20261005-124828`, gave ζ = 0.309).

| τ_Q (ms) | 20 | 50 | 100 | 200 | 400 | 770 |
|---|---|---|---|---|---|---|
| t̂ (ms), t_c from a_c = 91.078 (our a_c of this model) | 5.09 | 7.04 | 8.87 | 11.34 | 14.59 | 18.73 |
| X at freeze-out (µm) | 6.02 | 8.01 | 10.11 | 12.38 | 15.10 | 19.25 |

Freeze-out exponent ζ = 0.355 ± 0.004 (theirs: 0.346(2) quantum noise, 0.352(3)
thermal); with their quoted a_c ≈ 91.05 instead, ζ = 0.318 ± 0.003. The exponent
is very sensitive to the critical point: shifting a_c by 0.028 a0 moves t_c by
τ_Q × 0.0035 and ζ by 0.037, because t̂ ≪ τ_Q (a_c − a_f)/(a_i − a_f). In the
first-order regime, where t_c could be referred to a* or a_rot*, this will
matter. Correlation-length exponent 0.314 ± 0.005 (fit window 30–150 µm makes
no difference) against their 0.335(3) / 0.334(5): a remaining 3σ difference
(their τ_Q range starts at 1 ms; realizations 64 vs 400). Fitted modulation
period at freeze-out 2.42–2.54 µm (ground-state cell 2.69 µm). Cost: 0.68 ms per
step for 64 realizations; 26 min for the six ramps.

## 2026-10-05 — Error budget of the freeze-out exponents (`benchmarks/tube_ramp_errors.py`, run `20261005-205716_tube_ramp_errors`)

Median of per-realization crossings; bootstrap over realizations within each
τ_Q (1000 resamples); threshold f_th = 0.99, 0.98, 0.95, 0.90; reference shifted
by ±0.01 a0; grid term 0.015 (dx test). Total = quadrature sum of bootstrap,
half the threshold range, reference and grid terms.

| data set | ζ (f_th 0.98) | f_th 0.99 / 0.95 / 0.90 | bootstrap | reference | total |
|---|---|---|---|---|---|
| 6250/µm forward (from a_rot\*) | 0.342 | 0.347 / 0.334 / 0.329 | 0.005 | 0.007 | 0.019 |
| 2500/µm forward | 0.369 | 0.367 / 0.373 / 0.371 | 0.008 | 0.004 | 0.018 |
| 6250/µm reverse (from branch end) | 0.360 | 0.362 / 0.375 / 0.373 | 0.002 | 0.012 | 0.021 |
| 700/µm forward | 0.253 | 0.237 / 0.259 / 0.258 | 0.013 | 0.014 | 0.026 |

The threshold changes ζ by at most 0.018 and without a trend that would make
the exponent an artefact of the 0.98 criterion. With the full error the
first-order (0.342 ± 0.019) and continuous (0.369 ± 0.018) exponents differ by
~1σ and are consistent with each other and with 1/3; 700/µm stays clearly
lower. The bootstrap error is small because with quantum noise the
realizations differ little; systematic terms dominate.

## 2026-10-05 — Quasi-1D tube model (`dipgpe.quasi1d`, `benchmarks/quasi1d_validation.py`)

Reduced model with a variational transverse Gaussian (l, η), as in Blakie et al.
(arXiv:2004.12577) and Kirkby et al. (PRR 7, 2025). Roton instability of the
uniform tube with its own variational widths (runs `20261005-123604`, exact
Q5; `20261005-123934`, Q5 ≈ 1 + 3ε²/2 as in Blakie et al.), a0:

| n (/µm) | closed kernel + approx. LHY | Blakie 2020 | closed + exact Q5 | exact kernel + exact Q5 | full 3D (ours) |
|---|---|---|---|---|---|
| 700 | 81.221 | 81.2 | 80.579 | 80.256 | 84.50 |
| 2500 | 91.545 | a* ≈ 91.6 (continuous) | 90.563 | 90.030 | 92.32 |
| 6250 | 88.578 | 88.6 | 86.700 | 85.317 | 89.83 |

Modulated ground state at 88 a0, n = 2500/µm, minimizing over l, η and the cell
(run `20261005-123904`): closed kernel l = 1.078 µm, η = 4.248, cell 2.688 µm
(Kirkby et al.: l = 1.08 µm, η = 4.25, 344 µm / 128 cells = 2.69 µm); exact
kernel l = 1.080 µm, η = 4.191, cell 2.663 µm. With l, η fixed at Kirkby's values
the roton instability is at 91.08 a0 with their kernel (they report 91.05) and
90.544 a0 with the exact one (tests/test_quasi1d.py).

So dipgpe.quasi1d reproduces both papers exactly when given their ingredients. Two
approximations in those ingredients: the closed-form kernel is exact only for
η = 1 (checked against a direct transverse integral and against the 3D tube
code: exact kernel 1e-11, closed form 8.7e-6 off in the dipolar energy of a
modulated Gaussian tube at η = 4.25), and Q5 ≈ 1 + 3ε²/2 (5–6% below the exact
Q5 at ε ≈ 1.45–1.64). With exact ingredients the Gaussian ansatz is further from
the full 3D eGPE (−4.2, −2.3, −4.5 a0) than with both approximations (−3.3,
−0.8, −1.3 a0): the approximations partly cancel the error of the ansatz. The
reduced model also gives a larger first-order window a* − a_rot (2.1 and 1.0 a0
in Blakie et al.) than the full eGPE (0.78 and 0.43 a0). Consequence for C1:
quantitative statements in the first-order regime need the 3D model; the
reduced model is useful for large ensembles and for the n = 2500 reproduction
of Kirkby et al. (closed kernel), with 3D checks.

## 2026-10-04 — Discontinuous tube transitions (`uv run python benchmarks/tube_first_order.py --density N`)

Crystal branches followed upward in a_s by continuation (conjugate-gradient
ground states), per cell length L; a* where the crystal and uniform energies
per atom cross, maximized over L; the branch ends where the state relaxes to
the uniform tube. a_rot* from roton softening (tube_supersolid.py). Full eGPE:

| n (/µm) | a_rot* | a* | a* − a_rot* | contrast at a* | crystal branch ends | best cell | literature (reduced theory, verbatim) |
|---|---|---|---|---|---|---|---|
| 700 | 84.50 | 85.28 (L = 112 l); 85.32 for L → ∞ | +0.78 (+0.82) | 0.92 | 85.500–85.525 | grows without bound | a* = 83.3, a_rot = 81.2: discontinuous |
| 2500 | 92.32 | 92.30–92.325 (branch end, dE → 0⁻) | 0 | → 0 (0.08 at 92.30) | = a* | 2.79–2.88 µm | a* ≈ 91.6, continuous |
| 6250 | 89.83 | 90.256 | +0.43 | 0.71 | 90.375–90.400 | 2.76 µm (roton 2.56 µm) | a* = 89.6, a_rot = 88.6: discontinuous |

So at 700 and 6250/µm the crystal becomes the ground state before the roton
softens, with a finite contrast jump, and a metastable crystal branch persists
0.13–0.23 a0 above a* (the uniform tube is locally stable from a_rot* to a*):
first order with hysteresis on both sides. At 2500/µm the same procedure gives a
continuous transition at a_rot*. At 700/µm a*(L) = 85.324 − 4.67/L fits L = 64–112 l
to 2e-4 a0: near a* the optimal spacing diverges (isolated droplets, compact:
FWHM 0.80 µm at every L), consistent with the paper's direct BEC–droplet transition
at low density. Numerical uncertainty of a*: dx 0.25 → 0.125 ≤ 4e-4 a0, transverse
box ≤ 2e-3 a0, a_s interpolation < 1e-3 a0. Relative to the reduced theory the
full-eGPE a* is higher by +2.0 (700), +0.7 (2500), +0.65 (6250) a0, against the
literature's "approximately 1 a0 to 2 a0"; the gaps a* − a_rot* are smaller than
the reduced theory's (2.1 and 1.0 a0). Gate test: `tests/test_tube_first_order.py`
(n = 6250, 11 s). Scans: 2–6 min per density.


## Issues found and resolved

| # | Problem | Cause | Resolution | Status |
|---|---|---|---|---|
| C1-5 | Exponent fits and the hysteresis plot included onsets that happen after the ramp has stopped (found by an independent Codex reanalysis, 2026-10-06): at 6250/µm, τ_Q = 10 ms the median onset is 7.5 ms after the spinodal but the ramp ends 4.6 ms after it; at 700/µm, τ_Q = 10 and 20 ms likewise | The fits assumed every onset lies on the linear ramp, and the hysteresis point used a_rot* + rate × t̂, continuing the ramp past its end (86.84 a0 instead of the held 88.00 a0) | `tube_ramp_errors.py` fits only τ_Q whose median onset precedes the ramp end, uses one common window for the threshold test and adds the fit-window shift to the error; `tube_ramp_figures.py` places onsets on the actual ramp-and-hold schedule and marks post-ramp onsets open. 6250 forward ζ = 0.363 ± 0.021 (20–200 ms; 0.342 with the 10 ms point), 700/µm slope 0.22 (descriptive). Independently reproduced to all digits | fixed |
| C1-4 | Nucleation inside the window depends on the preparation, not only on (a_s, T): at the same ramp rate (τ_Q-equivalent 200 ms, 5 nK), a ramp starting at 92 a0 nucleates before the spinodal in 1/16 realizations, one starting at 90.5 a0 in 11/16; hold rates predict 0.25 | Thermal noise is sampled in the free-particle basis at a_i and the truncated-Wigner dynamics has no bath. The soft roton mode then carries a population set by the injection point and by adiabatic following (roton-band power at a*: 0.04 from 92 a0 slowly, 0.05 after the fast hold ramp, 0.1–0.2 from 90.5 a0), not the thermal kT/ω that grows towards the spinodal | Report Γ as the rate for the stated hold protocol, the ramp test from 92 a0 as the protocol-consistent check, and the preparation dependence as a limitation; quantitative Γ and τ_Q^× need a thermalizing c-field method (SPGPE) or Bogoliubov-basis thermal sampling | mitigated |
| C1-3 | Kirkby et al.'s thermal noise (App. A, T = 20 nK, modes below 2 k_BT, "3–5% excited atoms") cannot be reproduced as stated (finding) | With the standard free-particle energies k²/2 and orthonormal plane waves the expected added atoms are 15.5% of N (1D, 568 modes; 16.2% in 3D with transverse modes); their printed dispersion 2ħ²l²/(mL²) (no π²) gives 153% | `dipgpe.noise.wigner_noise` uses k²/2 and reports the added atoms. The n = 2500 anchor uses their quantum-noise-only variant (half a quantum per mode below 1/ξ, < 1% added), which they found gives the same exponents (ζ = 0.346 vs 0.352) | finding |
| C1-2 | Our reduced-model roton instability (80.3, 85.3 a0 at n = 700, 6250) disagreed with Blakie et al. 2020 (81.2, 88.6) (finding) | Their LHY uses Q5 ≈ 1 + 3ε²/2 and the closed-form kernel; we used the exact Q5 and kernel | With both of their approximations we get 81.221 and 88.578 a0 (and 91.545 at 2500 vs their a* ≈ 91.6); `quasi1d_validation.py --lhy approx` | finding |
| C1-1 | The quasi-1D dipolar kernel of Kirkby et al. (PRR 7, 2025, Eq. after (5)) does not match a direct integral over the transverse Gaussian for eta ≠ 1: up to ~2% at intermediate k for eta = 4.25 (finding) | As printed, with Ei(−u), it has the wrong k → ∞ limit; read with E1(u) = −Ei(−u) it has the right limits and is exact for eta = 1, but not for an elliptical cross-section | `dipgpe.quasi1d.dipolar_kernel_1d` is exact (1D quadrature, 1e-12 against the direct integral); `kernel="closed"` keeps their form. Roton instability at n = 2500/µm, l = 1.08 µm, eta = 4.25: 91.08 a0 with their form (they report 91.05) vs 90.544 a0 exact. Their conclusions (exponents) need not change, but a_c and the phase boundary of their reduced model shift by ~0.5 a0 | finding |
| BD-6 | q = 0 spectra of the tube crystal failed near the branch end (90.3 a0): `linalg.eigh` on a NaN Gram matrix, then CG for M⁻¹ not converging; at 90.0 and 90.2 a0 it worked but needed 600–1000 iterations (2026-10-06) | For a modulated state the translation mode ∂_zψ is a zero mode of M = L + 2X at q = 0, so M⁻¹ is singular; new LOBPCG directions with ~0 B-norm were rescaled by ~1e150 and overflowed, and the CG residual kept a component along ∂_zψ that its search directions could not remove | `bogoliubov(..., deflate=[∂_zψ])` solves the projected problem (initial block, preconditioned residuals and the CG residual projected off the deflated fields; the translation Goldstone pair is removed from the spectrum); directions with B-norm < 1e-14 of the block are dropped; Rayleigh–Ritz failures with P restart without P. Gate: deflating cos(k₁z) of a uniform gas removes exactly one ω(k₁) (test_bdg). Crystal at 90.3 a0: ω = 0.421 at q = 0 | fixed |

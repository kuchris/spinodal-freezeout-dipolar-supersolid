# Spinodal-controlled freeze-out and nucleation at the first-order superfluid–supersolid transition of a dipolar gas

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23181065.svg)](https://doi.org/10.5281/zenodo.23181065)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23181218.svg)](https://doi.org/10.5281/zenodo.23181218)

Code: Zenodo [10.5281/zenodo.23181065](https://doi.org/10.5281/zenodo.23181065) (all versions).
Reduced data: Zenodo dataset [10.5281/zenodo.23181218](https://doi.org/10.5281/zenodo.23181218)
(all versions; `data_C1.zip`, identical to `data_C1/` here).

Code, run records and analysis of the paper

> W. K. Wong, *Spinodal-controlled freeze-out and nucleation at the first-order
> superfluid–supersolid transition of a dipolar gas* (submitted; arXiv link to
> be added).

The simulations solve the three-dimensional extended Gross–Pitaevskii equation
for 164Dy in an infinite tube with truncated-Wigner noise, using the GPU solver
DipGPE included here (`src/dipgpe`, PyTorch; split-step dynamics, conjugate-
gradient ground states, Bogoliubov–de Gennes spectra, run records).

## Contents

| folder | content |
|---|---|
| `benchmarks/` | simulation and analysis scripts of the paper |
| `data_C1/` | per-realization observables of every simulation (NumPy archives, with parameters and checksums) |
| `runs/` | run records: command, parameters, code version, environment, logs and results of every computation |
| `docs/C1_record.md` | log of the results and of the issues found and resolved |
| `src/dipgpe/`, `tests/` | the solver and its tests |

## Install

    uv sync            # Python 3.13, PyTorch with CUDA 12.8 (see pyproject.toml)
    uv run python -m pytest -q

## Reproduce the paper

The data are included in `data_C1/` (85 MB: per-realization observables of
every simulation in the paper and the static results; see `data_C1/README.md`).
Run the analysis; each script writes a new record under `runs/`:

| result | command |
|---|---|
| exponents and error budget (Sec. III A, Appendix D) | `uv run python benchmarks/tube_ramp_errors.py` |
| Figures 1–7 (PNG and PDF in the new run record) | `uv run python benchmarks/tube_ramp_figures.py` |
| nucleation rates, ramp test (Sec. III C) | `uv run python benchmarks/tube_ramp_nucleation.py` |
| noise-cutoff sensitivity (Appendix C) | `uv run python benchmarks/tube_ramp_cutoff.py` |
| time-step check (Appendix B) | `uv run python benchmarks/tube_ramp_dt_check.py 20261006-142433` |
| thermal effects on the onset, Table V (Appendix F) | `uv run python benchmarks/tube_thermal_shift.py --holds 0:20261008-141000,5:20261008-133754,10:20261008-135021 --a-hold 90.8 --ramps-thermal 20:20261008-140146,50:20261005-175324+20261006-020234,100:20261005-175324+20261006-020234,200:20261008-140146 --ramps-quantum 20:20261005-132617,50:20261005-132617,100:20261005-133951,200:20261005-133951` |
| coexistence window and localized states (Appendix E) | values in `data_C1/static/*_tube_coexistence/tube_coexistence.json` and `*_tube_localized/tube_localized.json` |

The simulations themselves are rerun with `benchmarks/tube_ramp.py` (ramps
and holds; the parameters of every run are in `runs/<id>_tube_ramp_3d/meta.json`),
`tube_roton.py` and `tube_first_order.py` (spinodal and transition points),
`tube_crystal_bdg.py` (crystal soft mode), `tube_states.py` (Fig. 1d),
`tube_coexistence.py` (Maxwell construction) and `tube_localized.py` (localized
states); a production ramp takes 5–30 minutes on one RTX 5070 Ti.

## Provenance

The run records name the git commit of the development repository in which
each computation was made; this repository is a snapshot of it restricted to
this paper (see `RELEASE_SOURCE.txt`). The analysis scripts reproduce every
number of the paper from `data_C1/` alone.

## License and citation

MIT (see LICENSE). Please cite the paper; `CITATION.cff` describes this code
and data.

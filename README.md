# Spinodal-controlled freeze-out and nucleation at the first-order superfluid–supersolid transition of a dipolar gas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23181065.svg)](https://doi.org/10.5281/zenodo.23181065)

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
| `runs/` | run records: command, parameters, code version, environment, logs and results of every computation (fields in the data archive) |
| `docs/C1_record.md` | log of the results and of the issues found and resolved |
| `src/dipgpe/`, `tests/` | the solver and its tests |

## Install

    uv sync            # Python 3.13, PyTorch with CUDA 12.8 (see pyproject.toml)
    uv run python -m pytest -q

## Reproduce the paper

1. Download `data_C1.zip` from Zenodo ([DOI to be added]) and unzip it in the
   repository root, giving `data_C1/` (70 MB: per-realization observables of
   every simulation in the paper; see `data_C1/README.md`).
2. Run the analysis; each script writes a new record under `runs/`:

| result | command |
|---|---|
| exponents and error budget (Sec. III A, Appendix D) | `uv run python benchmarks/tube_ramp_errors.py` |
| Figures 1–6 (PNG and PDF in the new run record) | `uv run python benchmarks/tube_ramp_figures.py` |
| nucleation rates, ramp test (Sec. III C) | `uv run python benchmarks/tube_ramp_nucleation.py` |
| noise-cutoff sensitivity (Appendix C) | `uv run python benchmarks/tube_ramp_cutoff.py` |
| time-step check (Appendix B) | `uv run python benchmarks/tube_ramp_dt_check.py 20261006-142433` |

The simulations themselves are rerun with `benchmarks/tube_ramp.py` (ramps
and holds; the parameters of every run are in `runs/<id>_tube_ramp_3d/meta.json`),
`tube_roton.py` and `tube_first_order.py` (spinodal and transition points),
`tube_crystal_bdg.py` (crystal soft mode) and `tube_states.py` (Fig. 1d); a
production ramp takes 5–30 minutes on one RTX 5070 Ti.

## Provenance

The run records name the git commit of the development repository in which
each computation was made; this repository is a snapshot of it restricted to
this paper (see `RELEASE_SOURCE.txt`). The analysis scripts reproduce every
number of the paper from `data_C1/` alone.

## License and citation

MIT (see LICENSE). Please cite the paper; `CITATION.cff` describes this code
and data.

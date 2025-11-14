# Repository Guidelines

## Project Structure & Modules
- Core Python modules live in the repo root: `determinant_computer.py`, `fas_minor_calculator.py`, `output_display.py`, `hpc_fas_minor.py`, and `gpu_probe.py`.
- Tests are `test_*.py` files in the root (for example, `test_determinant_computer.py`, `test_display_widget.py`, `test_y_vector_quick.py`).
- Documentation is in `docs/`, example notebooks in `examples/`, and HPC batch scripts in `slurm/`.

## Setup, Build & Run
- Create a virtual environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run local calculations with `python fas_minor_calculator.py`.
- Probe GPU / environment with `python gpu_probe.py`.
- Use `python hpc_fas_minor.py` together with `slurm/*.sbatch` for cluster runs.

## Coding Style & Naming Conventions
- Target Python 3, with 4-space indentation and PEP 8–style formatting (black-compatible; default settings are fine if you use it).
- Functions and modules: `lower_snake_case`; classes: `CamelCase`; constants: `UPPER_SNAKE_CASE`.
- Prefer descriptive variable names; reserve single letters mainly for indices (`i`, `j`, `k`) in tight loops.

## Testing Guidelines
- Use `pytest` from the repo root: `pytest` or `pytest -k determinant`.
- Add new tests to `test_*.py`, mirroring the module under test.
- For numerical changes, include cases that check small symbolic matrices and at least one larger matrix for performance/precision regression.

## Commit & Pull Request Guidelines
- Write concise, present-tense commit messages (e.g., `add gpu probe utility`, `fix determinant sign handling`, `refactor hpc job submission`).
- For pull requests, include:
  - A short summary of the change and rationale.
  - Notes on any new CLI flags, dependencies, or HPC/GPU requirements.
  - How you tested the change (commands run, matrix sizes, and relevant test files).


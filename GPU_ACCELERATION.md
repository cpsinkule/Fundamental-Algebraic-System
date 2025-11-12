GPU Acceleration (Optional, Symbolic-Safe)

This file summarizes the optional GPU probing utilities that accelerate numeric evaluation while preserving the symbolic source of truth.

- Symbolic APIs remain authoritative (FASMinorCalculator, DeterminantComputer).
- GPU is used only to evaluate the same formulas numerically for fast exploration (for example, monomial search prefilters).

See README section "Optional GPU Probing" and gpu_probe.py for details and examples.

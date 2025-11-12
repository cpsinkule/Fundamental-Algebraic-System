# GPU-Accelerated Monomial Search (Prefilter + Exact Confirmation)

This guide shows how to quickly search a minor for a given monomial using the optional GPU probe, then confirm the result symbolically.

The symbolic APIs remain authoritative. The GPU step only evaluates the same formulas numerically to prefilter candidates.

## 1) Setup

```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer
from gpu_probe import GPUMinorProbe

# System
tuples = [(3, 1, 5), (3, 1, 4)]
calc = FASMinorCalculator.from_characteristic_tuples(tuples, use_symbolic=True)
det_comp = DeterminantComputer(calc)

# Extra row (graph_idx, vertex, layer)
extra_row = (0, 0, 1)

# GPU probe ('cupy' for NVIDIA CUDA if installed; otherwise 'numpy')
probe = GPUMinorProbe(calc, det_comp, backend='cupy')  # or 'numpy'
probe._prepare_for_extra_row(extra_row)
```

## 2) Define the target monomial

- Vertices: `('vertex', g, v)` → `u_{g,v}`
- Edges: `('edge', g, (src, tgt))` → `u_{g,(src,tgt)}`
- Values: non‑negative integer exponents

```python
mon = {
    ('vertex', 0, 0): 1,
    ('edge', 1, (0, 1)): 1,
}
```

## 3) GPU numeric prefilter

- Use `mode='divides'` to test whether the monomial divides some term (best for existence).
- Use `mode='exact'` to mask other `u` variables and test for the exact exponent pattern (heuristic).

```python
res_div = probe.probe_monomial_in_minor(extra_row, mon, mode='divides', samples=16)
res_exact = probe.probe_monomial_in_minor(extra_row, mon, mode='exact', samples=16)

print('Likely (divides):', res_div['likely'], 'samples:', res_div['samples'])
print('Likely (exact):  ', res_exact['likely'], 'samples:', res_exact['samples'])
```

Notes:
- `samples` (e.g., 16–32) increases confidence; results are probabilistic (Schwartz–Zippel style).
- If `backend='cupy'` is unavailable, set `backend='numpy'` for a CPU probe.

## 4) Exact symbolic confirmation (authoritative)

Once the probe says "likely", confirm with the symbolic APIs:

```python
# Exact coefficient (exact match)
coeff = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact', return_coeff=True)
print('Exact coefficient:', coeff)

# Residual polynomial (divides match)
residual = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='divides', return_coeff=True)
print('Residual (divides):', residual)
```

## 5) Batched probing (optional)

Evaluate many random assignments to keep the GPU busy:

```python
assignments = [probe.random_assignments(extra_row, seed=i) for i in range(16)]
vals = probe.evaluate_minor_numeric_batch(extra_row, assignments)
print('Batch minor values:', vals)
```

## Tips

- You do not need the symbolic minor expression to use the probe; just specify the same extra row you used.
- For large scans (many monomials), loop over monomial specs, use the GPU probe to prefilter, then call the exact symbolic method on positives.
- All exact results come from the existing symbolic APIs in `DeterminantComputer`.


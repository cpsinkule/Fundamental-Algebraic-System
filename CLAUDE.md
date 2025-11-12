# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python calculator for computing rows and minors of the **Fundamental Algebraic System (FAS)** principal part. It implements formulas from the Sinkule thesis on differential-algebraic systems, using symbolic computation (SymPy) with optional numeric GPU acceleration for monomial searches.

**Core concept**: Multi-component directed graphs → augmented matrix [A|b] → determinants/minors → DAE solvability analysis.

## Running Tests

```bash
# Run all tests
python -m pytest test_determinant_computer.py

# Run specific test class
python -m pytest test_determinant_computer.py::TestDeterminantComputerBasics

# Run single test
python -m pytest test_determinant_computer.py::TestDeterminantComputerBasics::test_initialization

# Verbose output
python -m pytest -v test_determinant_computer.py
```

## Dependencies

```bash
pip install -r requirements.txt  # Installs sympy>=1.9, numpy>=1.20.0
pip install cupy  # Optional, for GPU acceleration
```

## Architecture

### Three-Layer Design

1. **ComponentGraph** (`fas_minor_calculator.py`)
   - Represents directed graphs with local vertex numbering (always 0 to v-1)
   - Computes vertex/edge depths for the depth-based row selection algorithm
   - Factory method: `ComponentGraph.from_characteristic_tuple((r_0, ..., r_ω, v))`
   - Characteristic tuples define canonical graph structures with roots and layers

2. **FASMinorCalculator** (`fas_minor_calculator.py`)
   - Main calculator: constructs symbolic [A|b] rows on-demand
   - Takes list of ComponentGraph instances (need ≥2 for non-trivial systems)
   - Implements recursive formulas from thesis (equations 5.16-5.17 for A, 6.4/6.14 for b)
   - Key method: `get_row(graph_idx, vertex, layer)` → returns SymPy Matrix row
   - **IMPORTANT**: Layers start at 1, not 0 (A^1, A^2, ...)
   - Performance: lazy structure function loading, caching, simplification thresholds

3. **DeterminantComputer** (`determinant_computer.py`)
   - Interface for computing minors: automatically generates n-m base rows via depth algorithm
   - User provides just ONE extra row → computes (n-m+1)×(n-m+1) determinant
   - Two paths: `compute_minor()` (standard) and `compute_minor_fast()` (block expansion)
   - Monomial search: `find_monomial_in_minor()`, `coeff_of_monomial()`
   - Root product extraction: `base_A_root_product_spec()`, `coeff_of_base_A_root_product()`

4. **GPUMinorProbe** (`gpu_probe.py`) [Optional]
   - Numeric prefilter: evaluates minors on GPU without symbolic expansion
   - Use case: quickly test if a monomial exists before expensive symbolic confirmation
   - Backend: 'cupy' (GPU) or 'numpy' (CPU fallback)
   - Does NOT replace symbolic computation—remains source of truth

### Data Flow for Computing a Minor

```
Characteristic Tuples → ComponentGraph.from_characteristic_tuple()
                     ↓
               ComponentGraph instances
                     ↓
          FASMinorCalculator([graph1, graph2, ...])
                     ↓
          DeterminantComputer(calc)
          - Auto-generates n-m base rows using depth algorithm
                     ↓
          User specifies ONE extra row: (graph_idx, vertex, layer)
                     ↓
          compute_minor() or compute_minor_fast()
          - Retrieves base rows via calc.get_row()
          - Retrieves extra row via calc.get_row()
          - Assembles matrix, computes determinant
                     ↓
               Symbolic minor (SymPy expression)
```

### Key Formulas and Structure

**Matrix structure:**
- **n** = total_vertices + total_edges (manifold dimension)
- **m** = total_vertices (number of constraints)
- Augmented system [A|b] has **n-m+1 columns** (one per edge + b column)
- Need **n-m+1 rows** for square determinant

**Row recursion (A entries, equation 5.16-5.17):**
```
a^{s+1}_{v,w} = h₁(a^s_{v,w}) = principal part of applying Hamiltonian vector field
              = Σ_{l∈E(Γ)} a^s_{v,l} · q_{l,w}
where q_{l,w} = Σ_{i∈V(Γ)} c^l_{i,w} · u_i
```
- **Principal part**: degree 1 in vertex variables
- A entries are extracted via `_extract_principal_part(expr, 1)`

**B vector (equations 6.4 and 6.14):**
```
Layer 1: b^1_v = degree-2 terms from specific formula
Layer s>1: b^s_v = h₂(b^{s-1}_v) + correction terms
```
- **Principal part**: degree 2 in vertex variables
- Often zero due to missing Type 3 structure functions (c^l_{w,v} with vertex superscripts)

**Structure functions:**
- Type 1: c^k_{i,j} (k=edge, i,j=vertices) - fully implemented with 6 Tanaka constraints
- Type 2: c^k_{l,i} (k,l=edges, i=vertex) - implemented
- Type 3: c^l_{w,v} (l,w,v=vertices) - documented but NOT implemented → b often zero

**Depth-based base row selection:**
For each component with ω = num_roots - 1:
- Layer s=1: select vertices with depth ≥ 1
- Layer s=2: select vertices with depth ≥ 2
- ...
- Layer s=ω+1: select vertices with depth ≥ ω+1
This produces exactly n-m rows automatically.

## Critical Implementation Details

### Vertex Numbering
- **Always local within each component**: vertices numbered 0 to v-1
- NOT global numbering across components (despite some old documentation suggesting otherwise)
- Rows specified as `(graph_idx, local_vertex, layer)`
- Example: Component 0 has vertices [0,1,2], Component 1 has vertices [0,1,2] (separate)

### Layer Numbering
- **Layers start at 1**: A^1, A^2, A^3, ... (NOT A^0)
- When specifying rows: `(graph_idx, vertex, layer)` with `layer >= 1`

### Depth Computation
```python
# Root vertices (first ω+1 vertices): depth(root_i) = i
# Edges: depth(edge) = depth(source_vertex)
# Non-root vertices: depth(v) = max(edge_depths) + 1  # NOTE: +1 is applied
```
**WARNING**: Code adds +1 but some documentation examples don't reflect this. Code is correct.

### Performance Flags
```python
FASMinorCalculator(
    graphs,
    use_symbolic=True,
    use_lazy_structure_functions=True,  # Default: lazy load (~75% memory savings)
    enable_simplification=True,          # Default: varies
    simplification_threshold=10000       # Simplify expressions above this length
)
```

### Structure Function Constraints (Type 1)
Six constraints implemented for c^k_{i,j} (k=edge, i,j=vertices):
0. Diagonal zero: i ≠ j
1. Component locality: all in same component
2. Edge definition: k must connect i or j
3. Depth constraint: depth(i) ≤ depth(k)
4. Depth ordering: depth(i) < depth(j)
5. Root exclusion: j cannot be a root

See `CONSTRAINTS.md` for full details and mathematical justification.

## Common Workflows

### Compute a Single Minor
```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

# From characteristic tuples
calc = FASMinorCalculator.from_characteristic_tuples(
    [(3, 1, 5), (3, 1, 4)],  # Two components
    use_symbolic=True
)
det_comp = DeterminantComputer(calc)

# Compute minor (base rows auto-generated + one extra row)
minor = det_comp.compute_minor_fast(0, 0, 1)  # graph_idx=0, vertex=0, layer=1
```

### Search for Monomials in Minor
```python
# Define monomial: u_{0,0} * u_{1,(0,1)}
mon = {
    ('vertex', 0, 0): 1,        # graph 0, vertex 0, exponent 1
    ('edge', 1, (0, 1)): 1      # graph 1, edge (0,1), exponent 1
}

# Exact coefficient
coeff = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact')

# Check if monomial divides any terms (returns residual polynomial)
residual = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='divides')
```

### GPU Numeric Prefilter (Optional)
```python
from gpu_probe import GPUMinorProbe

probe = GPUMinorProbe(calc, det_comp, backend='cupy')
extra_row = (0, 0, 1)
probe._prepare_for_extra_row(extra_row)

# Numeric test (fast): does monomial likely exist?
result = probe.probe_monomial_in_minor(extra_row, mon, mode='divides', samples=32)
if result['likely']:
    # Confirm with exact symbolic extraction
    coeff = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact')
```

### Inspect Base Rows
```python
base_rows = det_comp.get_base_rows()
print(f"Auto-generated {len(base_rows)} base rows:")
for g, v, s in base_rows:
    print(f"  Component {g}, vertex {v}, layer {s}")
```

## Documentation Structure

- `README.md`: User-facing guide with examples and API reference
- `IMPORTANT_NOTES.md`: Critical corrections (layers start at 1, n=vertices+edges, multiple components required)
- `CONSTRAINTS.md`: Complete structure function constraint reference (Types 1-3)
- `DEPTH_IMPLEMENTATION.md`: Depth computation algorithm details
- `CORRECTIONS_FROM_THESIS.md`: Fixes to match Sinkule thesis formulas
- `B_VECTOR_IMPLEMENTATION.md`: b vector equations 6.4/6.14, why b is often zero
- `PERFORMANCE_NOTES.md`: Lazy loading, simplification thresholds, benchmarks
- `INDEX_ORDERING_FIX.md`: Historical fix for c^k_{i,j} vs c^k_{j,i}
- `GPU_ACCELERATION.md`: Brief note on optional GPU utilities
- `docs/MONOMIAL_SEARCH_GPU.md`: Detailed GPU monomial search guide

## Known Limitations

1. **Type 3 structure functions not implemented**: c^l_{w,v} (vertex superscripts) are documented but not created by the calculator, causing b vector entries to often be zero. This is expected per current implementation.

2. **Symbolic expressions grow quickly**: Large systems (many components, deep layers) produce massive SymPy expressions. Use `compute_minor_fast()` and GPU probing to mitigate.

3. **No numeric-only mode**: Calculator is symbolic-first. GPU utilities only provide numeric prefilters, not standalone numeric computation.

## Testing Philosophy

- Test suite focuses on `DeterminantComputer` (26 tests in `test_determinant_computer.py`)
- Tests cover: initialization, validation, base row generation, minor computation, monomial search
- Multiple test classes: Basics, Validation, Integration, BaseRows, ComputeMinor, Examples
- Documentation references several test files (test_depth.py, test_b_vector.py) that don't currently exist

## Code Style Notes

- Type hints used throughout
- SymPy for all symbolic computation
- Caching via `@lru_cache` or manual dictionaries for expensive operations
- Local vertex numbering enforced (0 to v-1 within each component)
- Comments reference thesis equation numbers (e.g., "equation 5.16")

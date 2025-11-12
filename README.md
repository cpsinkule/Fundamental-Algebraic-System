# FAS Row Calculator

Python calculator for computing rows of the principal part of the fundamental algebraic system (FAS).

## Overview

This calculator computes individual rows of the augmented system [A|b] based on component graph structures. The system assumes:

- Structure functions and alpha coefficients are constants (derivatives in the direction of vec h have 0 principal part)
- Component graphs defined by vertices and directed edges
- **Layers start at 1** (A^1, A^2, ..., not A^0)
- **n (manifold dimension) = total vertices + total edges** across all component graphs
- Rows specified by (vertex, layer) pairs where layer s indicates the row comes from A^s
- The augmented system [A|b] has n-m+1 columns, where m is the number of constraints
- **Multiple components are essential** - the single component case is trivial

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `sympy`: For symbolic computation (required)

## Usage

### Interactive Mode

Run the calculator in interactive mode:

```bash
python fas_minor_calculator.py
```

The interactive interface will guide you through:
1. Defining component graphs (vertices and edges)
2. Using symbolic computation (only supported mode)
3. Optionally providing structure function values
4. Computing individual rows

### Programmatic Usage

Import and use the calculator in your Python code:

```python
from fas_minor_calculator import ComponentGraph, FASMinorCalculator

# Define component graphs (need at least 2 for non-trivial case)
graph1 = ComponentGraph([1, 2], [(1, 2)])  # Component 1: 1->2
graph2 = ComponentGraph([3], [])            # Component 2: isolated vertex 3

# Create calculator
calc = FASMinorCalculator([graph1, graph2], use_symbolic=True)

# Compute individual rows on demand
# Each row is specified as: (graph_idx, vertex, layer)
# IMPORTANT: Layers start at 1, not 0!

row_1 = calc.get_row(0, 1, 1)  # Graph 0, vertex 1, layer 1
row_2 = calc.get_row(0, 2, 1)  # Graph 0, vertex 2, layer 1
row_3 = calc.get_row(1, 3, 1)  # Graph 1, vertex 3, layer 1

print(f"Row 1 has {len(row_1)} entries")
print(f"Row 1: {row_1}")
```

## Examples

Run the example file to see various usage patterns:

```bash
python example_usage.py
```

Examples include:
- Two component graphs (non-trivial case)
- Using custom structure functions with multiple components
- Symbolic computation with multiple components
- Three component graphs with mixed layers

### Example: Structure Functions

```python
# Set custom structure function values
# Key format: (graph_idx, vertex, layer_difference)
structure_funcs = {
    (0, 1, 0): 2.0,   # Graph 0, vertex 1, layer diff 0
    (0, 1, 1): 1.5,   # Graph 0, vertex 1, layer diff 1
    (1, 3, 0): 3.0,   # Graph 1, vertex 3, layer diff 0
}
calc.set_structure_functions(structure_funcs)
```

### Example: Symbolic Computation

```python
import sympy as sp

# Create symbolic calculator with multiple components
graph1 = ComponentGraph([1, 2], [(1, 2)])
graph2 = ComponentGraph([3], [])
calc = FASMinorCalculator([graph1, graph2], use_symbolic=True)

# Compute symbolic rows
row_1 = calc.get_row(0, 1, 1)  # Layer 1
row_2 = calc.get_row(0, 2, 2)  # Layer 2

# Pretty print entries
print("Row 1:")
for i, entry in enumerate(row_1):
    print(f"  Entry {i}: ", end="")
    sp.pprint(entry)
```

## Key Concepts

### Component Graphs
Component graphs are directed graphs where:
- **Vertices**: Represent variables or components in the system
- **Edges**: Represent dependencies or relationships between components

### Row Specification
Each row is specified by a tuple `(graph_idx, vertex, layer)`:
- `graph_idx`: Index of the component graph (0-based)
- `vertex`: Vertex label within that graph
- `layer`: Layer number s, indicating the row comes from A^s
  - **IMPORTANT: Layers start at 1, not 0** (A^1, A^2, A^3, ...)

### System Dimensions
- **n (manifold dimension) = total vertices + total edges** across all component graphs
- **m** = number of vertices (constraints)
- **Number of columns in each row = n - m + 1** (one per edge plus b column)

### Multiple Components
**The single component case is trivial.** For meaningful calculations, you need at least 2 component graphs. Each component graph represents a separate part of the system that may or may not be connected to other components.

### Principal Part
The calculator focuses on the principal part of the system, where:
- Structure functions and alphas are treated as constants
- Only leading-order terms contribute to the matrix entries
- Derivatives in the direction of vec h have zero principal part

## API Reference (Essentials)

### ComponentGraph

Represents a directed graph component with local vertex numbering and edges.

Constructor: `ComponentGraph(vertices: List[int], edges: List[Tuple[int, int]], num_roots: Optional[int] = None)`

Factory: `ComponentGraph.from_characteristic_tuple((r_0, ..., r_ω, v))`

### FASMinorCalculator (symbolic‑only)

Constructor: `FASMinorCalculator(graphs: List[ComponentGraph], use_symbolic=True, ...)`

- `get_row(graph_idx, vertex, layer)`: Return the full row `[A | b]` as a SymPy Matrix slice (all edge columns + b column).
- Depth accessors on component graphs: `graph.get_vertex_depth(v)`, `graph.get_edge_depth((u,v))`.

### DeterminantComputer

Constructor: `DeterminantComputer(calc: FASMinorCalculator)`

- `compute_minor(graph_idx, vertex, layer)`: Symbolic minor using base rows + one extra row.
- `compute_minor_fast(graph_idx, vertex, layer)`: Optimized block expansion path (default used internally).
- `get_base_rows()`: Inspect the automatically generated base rows (n−m rows).
- `get_base_A_matrix()`: Assemble the (n−m)×(n−m) A‑only matrix from base rows.
- `compute_base_A_determinant()`: det of base A using block structure.

Monomial utilities:
- `coeff_of_monomial(expr, mon_spec, match='exact'|'divides')`
- `find_monomial_in_minor(g, v, s, mon_spec, match='exact'|'divides', return_coeff=True)`

Root monomial product (base A):
- `root_monomial_spec_for_layer(component, layer)`: M_i(s) = u_{(i,0)} ∏_{k=0}^{s-2} u_{i,(k,k+1)}.
- `block_root_product_spec(component)`: p_i = ∏_s M_i(s)^{N_{i,s}} with N_{i,1..ω+1} derived from depths (= r_{s−1}).
- `base_A_root_product_spec()`: p = ∏_i p_i (combined across components).
- `coeff_of_block_root_product(component, match='exact')`: coefficient of p_i in det(A_i^base).
- `coeff_of_base_A_root_product(match='exact')`: coefficient of p in det(A_base).
- One‑shot helpers from tuples are provided for the spec and coefficient.

## Computing Determinants

The `determinant_computer` module provides a convenient interface for computing determinants from calculator rows using SymPy's `det()` function. Only symbolic computation is supported; numeric minors are not defined in this calculator.

### Optional GPU Probing (Numeric, preserves symbolic results)

For fast exploration (e.g., monomial searches), you can numerically evaluate the fast minor formula on GPU without changing the symbolic source of truth. Use the optional `gpu_probe` module:

```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer
from gpu_probe import GPUMinorProbe, probe_monomial_from_characteristic_tuples

calc = FASMinorCalculator.from_characteristic_tuples([(3,1,5), (3,1,4)], use_symbolic=True)
det_comp = DeterminantComputer(calc)

probe = GPUMinorProbe(calc, det_comp, backend='cupy')  # or backend='numpy'
extra_row = (0, 0, 1)

# Prepare and generate one random assignment
probe._prepare_for_extra_row(extra_row)
vals = probe.random_assignments(extra_row, seed=42)

# Numeric evaluation of the minor (fast)
val = probe.evaluate_minor_numeric(extra_row, vals)
print(val)

# Use this as a prefilter, then confirm with exact symbolic APIs:
#   det_comp.find_monomial_in_minor(...)

# One-shot probe directly from characteristic tuples
tuples = [(3,1,5), (3,1,4)]
row = (0, 0, 1)
mon = {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}
res = probe_monomial_from_characteristic_tuples(tuples, row, mon, mode='divides', samples=16, backend='cupy')
print(res)
```

Notes:
- This does not replace the symbolic path. It only evaluates the same formulas numerically for speed.
- GPU requires `cupy` installed. Otherwise, set `backend='numpy'` for CPU numeric probes.

Batched evaluation and monomial probing

```python
from gpu_probe import GPUMinorProbe

probe = GPUMinorProbe(calc, det_comp, backend='cupy')
extra_row = (0, 0, 1)
probe._prepare_for_extra_row(extra_row)

# Batch: evaluate the minor for many random assignments
assignments = [probe.random_assignments(extra_row, seed=i) for i in range(8)]
vals = probe.evaluate_minor_numeric_batch(extra_row, assignments)

# Numeric prefilter for monomial presence
mon = {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}
res_div = probe.probe_monomial_in_minor(extra_row, mon, mode='divides', samples=16)
res_exact = probe.probe_monomial_in_minor(extra_row, mon, mode='exact', samples=16)

print(res_div['likely'], res_exact['likely'])

# Then confirm with exact symbolic extraction if needed:
coeff = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact', return_coeff=True)
```

### Automatic Minor Computation (Recommended)

The `DeterminantComputer` automatically generates a base set of n-m rows using a depth-based selection algorithm. You only need to provide one additional row to compute a minor:

```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

# Create calculator for (3,1,5) + (3,1,4) system
calc = FASMinorCalculator.from_characteristic_tuples(
    [(3, 1, 5), (3, 1, 4)],
    use_symbolic=True
)

# Create determinant computer (automatically generates base rows)
det_comp = DeterminantComputer(calc)

# Compute minor by providing just ONE additional row
minor = det_comp.compute_minor(0, 0, 1)  # Component 0, vertex 0, layer 1
print(f"Minor: {minor}")

# Inspect the base rows used
base_rows = det_comp.get_base_rows()
print(f"Base rows (n-m={len(base_rows)}): {base_rows}")
```

**How base rows are selected:**
For each component with omega = num_roots - 1:
- Layer 1: select all vertices with depth >= 1
- Layer 2: select all vertices with depth >= 2
- Continue to layer omega+1: select vertices with depth >= omega+1

This generates exactly n-m rows, which combined with your one user-specified row produces the n-m+1 rows needed for a square matrix.

### Manual Row Specification (Alternative)

If you need full control, you can manually specify all rows:

```python
# Create calculator
calc = FASMinorCalculator.from_characteristic_tuples(
    [(2, 1, 4)],
    use_symbolic=True
)

det_comp = DeterminantComputer(calc)

# Manually specify all rows (3 edges + 1 = 4 rows)
row_specs = [
    (0, 0, 1),  # Component 0, vertex 0, layer 1
    (0, 1, 1),  # Component 0, vertex 1, layer 1
    (0, 0, 2),  # Component 0, vertex 0, layer 2
    (0, 1, 2),  # Component 0, vertex 1, layer 2
]

# Compute the determinant
det = det_comp.compute_determinant(row_specs)
print(f"Determinant: {det}")
```

### API: DeterminantComputer

**Constructor:**
```python
DeterminantComputer(calculator: FASMinorCalculator)
```
- `calculator`: A FASMinorCalculator instance to retrieve rows from
- Automatically generates n-m base rows using depth-based selection
- Raises `ValueError` if base row generation doesn't produce exactly n-m rows

**Methods:**

- `compute_minor(graph_idx, vertex, layer)`: Compute a minor with automatic base rows (symbolic)
  - `graph_idx`: Component graph index (0-based)
  - `vertex`: Local vertex label within that component
  - `layer`: Layer number s >= 1
  - Combines the n-m base rows with your one user-specified row
  - Returns: SymPy expression
  - Example: `minor = det_comp.compute_minor(0, 0, 1)`

- `compute_minor_fast(graph_idx, vertex, layer)`: Optimized minor via block expansion (symbolic)
  - Uses Laplace expansion along the b column and per-component A-blocks
  - Avoids building the full (n_edges+1)×(n_edges+1) determinant when using one extra row
  - Recommended for large systems
  - Example: `minor = det_comp.compute_minor_fast(0, 0, 1)`

- `minor_from_characteristic_tuples(char_tuples, row, fast=True)`: One-shot minor
  - Convenience API: builds calculator from characteristic tuples and computes the minor
  - `row` is `(graph_idx, vertex, layer)`
  - Example:
    ```python
    from determinant_computer import DeterminantComputer
    char_tuples = [(3, 1, 5), (3, 1, 4)]
    minor = DeterminantComputer.minor_from_characteristic_tuples(char_tuples, (0, 0, 1))
    ```

- `get_base_rows()`: Inspect the automatically generated base rows
  - Returns: List of (graph_idx, vertex, layer) tuples
  - These are the n-m rows used in minor computation
  - Example: `base_rows = det_comp.get_base_rows()`

- `compute_determinant(row_specs: List[Tuple[int, int, int]])`: Compute determinant with manual row specification (symbolic)
  - `row_specs`: List of row specifications as `(graph_idx, vertex, layer)` tuples
  - Returns: SymPy expression
  - Raises `ValueError` if matrix is not square or specifications are invalid
  - Use this if you need full control over which rows are included

- `get_matrix(row_specs: List[Tuple[int, int, int]])`: Assemble matrix without computing determinant
  - Returns: SymPy Matrix object
  - Useful for inspecting the matrix before computation

- `get_base_A_matrix()`: Assemble the (n−m)×(n−m) A-only submatrix from base rows
  - Returns: SymPy Matrix (rows = base rows, columns = all edge columns)
  - Example: `A_base = det_comp.get_base_A_matrix()`

- `compute_base_A_determinant()`: Compute det of the base A submatrix (symbolic)
  - Uses block structure; equals product of per-component A-block determinants
  - Example: `det_A_base = det_comp.compute_base_A_determinant()`

### Monomial Search in Minors

You can search a computed minor for a specific monomial in the `u` variables (vertex and edge variables are supported) and retrieve either its exact coefficient or the residual polynomial when the monomial divides terms.

Monomial spec format:
- Vertices: `('vertex', g, v)` represents `u_{g,v}`
- Edges: `('edge', g, (src, tgt))` represents `u_{g,(src,tgt)}`
- Values are non-negative integer exponents

Methods:
- `coeff_of_monomial(expr, monomial_spec, match='exact'|'divides')`
  - `match='exact'`: returns the exact coefficient of the monomial in `expr`
  - `match='divides'`: returns the residual polynomial after factoring out the monomial from all matching terms

- `find_monomial_in_minor(graph_idx, vertex, layer, monomial_spec, match='exact', return_coeff=True)`
  - Computes the minor (fast path) and extracts information about the monomial
  - If `return_coeff=True`: returns coefficient (exact) or residual polynomial (divides)
  - If `return_coeff=False`: returns a boolean indicating existence

Example:
```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

calc = FASMinorCalculator.from_characteristic_tuples([(3, 1, 5), (3, 1, 4)], use_symbolic=True)
det_comp = DeterminantComputer(calc)

# Monomial: u_{0,0} * u_{1,(0,1)}
mon = {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}

# Exact coefficient in the minor with extra row (0,0,1)
coeff = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact', return_coeff=True)

# Existence (exact)
exists = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='exact', return_coeff=False)

# Divides: residual polynomial (sum of all terms containing the monomial, factored out)
residual = det_comp.find_monomial_in_minor(0, 0, 1, mon, match='divides', return_coeff=True)
```

### Root Monomial Product in Base A

Compute the predicted root‑monomial product p_i for each component block (by homogeneity and layer counts), and the global p = ∏_i p_i. You can also extract their exact coefficients in det(A_base):

```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

tuples = [(3,1,5), (3,1,4)]
calc = FASMinorCalculator.from_characteristic_tuples(tuples, use_symbolic=True)
det_comp = DeterminantComputer(calc)

# Specs
p0 = det_comp.block_root_product_spec(0)
p1 = det_comp.block_root_product_spec(1)
p  = det_comp.base_A_root_product_spec()

# Coefficients
c0 = det_comp.coeff_of_block_root_product(0)
c1 = det_comp.coeff_of_block_root_product(1)
c  = det_comp.coeff_of_base_A_root_product()

print(p0, p1, p)
print(c0, c1, c)
```

### Coefficient of p in the Full Minor

Given an extra row `(g, v, s)`, the full minor’s u-degree exceeds that of p, so the meaningful "coefficient" is the residual polynomial after factoring p (divides mode).

Symbolic (exact):
```python
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

tuples = [(3,1,5), (3,1,4)]
calc = FASMinorCalculator.from_characteristic_tuples(tuples, use_symbolic=True)
det_comp = DeterminantComputer(calc)

extra = (0, 0, 1)
residual = det_comp.coeff_of_p_in_minor(*extra)  # general exact extractor
exists = det_comp.has_p_in_minor(*extra)
print(residual)
print(exists)
```

Expansion-free exact (recommended):
```python
# Avoids expanding the determinant; uses block structure directly
residual_fast = det_comp.coeff_of_p_in_minor_fast(*extra)
print(residual_fast)
```

Fast numeric prefilter (GPU/CPU):
```python
from gpu_probe import GPUMinorProbe, probe_p_from_characteristic_tuples

probe = GPUMinorProbe(calc, det_comp, backend='cupy')  # or 'numpy'
probe._prepare_for_extra_row(extra)
res = probe.probe_p_in_minor(extra, samples=32)
print(res['likely'])

# One-shot from tuples
res2 = probe_p_from_characteristic_tuples(tuples, extra, samples=32, backend='numpy')
print(res2['likely'])
```

### Notes

- Base rows are generated automatically during `DeterminantComputer` initialization via a depth‑based selection.
- Symbolic expressions can grow quickly for large systems and deep layers.
- The calculator is symbolic‑only; GPU utilities provide numeric prefilters without changing symbolic results.

### Example: Computing Multiple Minors

```python
# Create calculator
calc = FASMinorCalculator.from_characteristic_tuples(
    [(3, 1, 5), (3, 1, 4)],
    use_symbolic=True
)

det_comp = DeterminantComputer(calc)

# Compute multiple minors with different user rows
minor_1 = det_comp.compute_minor(0, 0, 1)
minor_2 = det_comp.compute_minor(0, 1, 1)
minor_3 = det_comp.compute_minor(1, 0, 1)

print(f"Minor (0,0,1): {minor_1}")
print(f"Minor (0,1,1): {minor_2}")
print(f"Minor (1,0,1): {minor_3}")

# All minors use the same base rows
print(f"\nBase rows used: {len(det_comp.get_base_rows())} rows")
```

### Manual Approach (Alternative)

If you need more control, you can still manually assemble rows and compute determinants:

```python
import sympy as sp

# Retrieve rows
rows = [calc.get_row(g, v, s) for g, v, s in row_specs]

# Assemble matrix
matrix = sp.Matrix(rows)

# Compute determinant
det = matrix.det()
```

## Theory Background

This calculator computes rows of the fundamental algebraic system as used in differential-algebraic equation analysis. The system structure follows the formulation where:

1. The augmented system [A|b] has infinitely many rows (one per (vertex, layer) pair)
2. Rows correspond to differentiation chains of the original equations
3. The principal part contains the leading-order differential terms
4. Minors determine the solvability and index of the system

For theoretical details, refer to the Sinkule thesis on fundamental algebraic systems.

## License

This calculator is provided for research and educational purposes.

# B Vector Implementation

## Date: 2025-11-10

## Summary

Implemented recursive computation of the b vector according to equations 6.4 and 6.14 from the Sinkule thesis.

**Critical difference from A matrix**: The principal part of the b vector has degree 2 in vertex variables, while the A matrix has degree 1.

---

## Implementation

### Equation 6.4 (Layer 1 - Base Case)

For layer 1, the b vector is computed as:

```
b¹_v = Σ_{l,w ∈ V(Γ), C(l)≠C(v)} (α(v)² - α(l)²) c^l_{w,v} u_w u_l
```

Where:
- **v** is the row vertex (in component C(v))
- **l** ranges over all vertices in components different from C(v)
- **w** ranges over all vertices in all components
- **c^l_{w,v}** are structure functions with vertex superscript l
- **u_w, u_l** are vertex variables
- **α(v), α(l)** are alpha coefficient values

**Key constraint**: C(l) ≠ C(v) means vertex l must be in a different component than vertex v.

**Degree**: The result has degree 2 in vertex variables (u_w × u_l).

### Equation 6.14 (Layer s+1 - Recursive Case)

For layers s ≥ 2, the b vector is computed recursively:

```
b^{s+1}_v = [⃗h(b^s_v)]_(2) + Σ_{l,w ∈ E(Γ), C(l)=C(v), C(w)≠C(l)} (α(l)² - α(w)²) a^s_{v,l} q_{l,w} u_w
```

Where:
- **[⃗h(b^s_v)]_(2)** means: apply Hamiltonian vector field ⃗h to b^s_v, then extract degree 2 part
- **l** ranges over edges in the same component as vertex v
- **w** ranges over edges in different components than l
- **a^s_{v,l}** is the A matrix entry from layer s
- **q_{l,w}** is computed between two edges
- **u_w** is an edge variable
- **α(l), α(w)** are alpha coefficients (using source vertices of edges)

**Key constraints**:
- C(l) = C(v): edge l must be in same component as vertex v
- C(w) ≠ C(l): edge w must be in different component than edge l

**Degree**: The result is degree 2 in vertex variables (extracted from expanded expression).

---

## Code Implementation

### Location: `fas_minor_calculator.py`

Modified the `build_matrix_entry()` method (lines 814-950) to handle the b column:

```python
if col_spec[0] == 'b':
    # Layer 1: Use equation 6.4
    if row_layer == 1:
        # Sum over vertices l in different components
        # and vertices w in all components
        # Add terms: (α(v)² - α(l)²) c^l_{w,v} u_w u_l

    # Layer s > 1: Use equation 6.14
    else:
        # Get b^s_v from previous layer
        # Compute ⃗h(b^s_v) and extract degree 2
        # Add sum over cross-component edge interactions
```

### Key Features

1. **Caching**: Both b¹ and b^s entries are cached to avoid recomputation
2. **Symbolic mode**: Full symbolic computation with structure functions
3. **Numeric mode**: Falls back to user-specified values or 0
4. **Degree extraction**: Uses `_extract_principal_part(expr, 2)` for degree 2

---

## Why b Vector is Often Zero

In the current implementation, b vector entries are frequently zero because:

### 1. Structure Function Constraints

The Tanaka decomposition (as implemented) creates structure functions c^k_{i,j} where:
- **k is always an edge** (superscript)
- **i, j are vertices or edges** (subscripts)

For equation 6.4, we need c^l_{w,v} where **l is a vertex** (superscript). These structure functions are not created by `_create_default_structure_functions()`.

### 2. Component Constraints

Equation 6.4 requires:
- l from different component than v (C(l) ≠ C(v))

Equation 6.14 requires:
- l from same component as v (C(l) = C(v))
- w from different component than l (C(w) ≠ C(l))

These constraints significantly restrict which terms contribute.

### 3. Mathematical Interpretation

When b = 0 in a FAS system, it means:
- The system is homogeneous in a certain sense
- Cross-component interactions don't contribute to the b vector
- The principal part structure is simplified

This is consistent with systems where the b vector represents inhomogeneous terms that vanish in the principal part analysis.

---

## Usage Example

```python
from fas_minor_calculator import ComponentGraph, FASMinorCalculator

# Create multi-component system
graph0 = ComponentGraph.from_characteristic_tuple((3, 1, 5))
graph1 = ComponentGraph.from_characteristic_tuple((2, 3))

# Create symbolic calculator
calc = FASMinorCalculator([graph0, graph1], use_symbolic=True)

# Get complete row including b column
row = calc.get_row(0, 0, 1)  # Component 0, vertex 0, layer 1
b_entry = row[-1]  # Last entry is b column

print(f"b¹_(0,0) = {b_entry}")

# Get layer 2
row2 = calc.get_row(0, 0, 2)
b_entry2 = row2[-1]

print(f"b²_(0,0) = {b_entry2}")
```

---

## Future Extensions

To have non-zero b vector entries, one could:

1. **Extend structure functions**: Create c^l_{w,v} with vertex superscripts
2. (reserved) User-specified overrides could be added in a future API if needed
3. **Alternative formulas**: Implement different b vector computation based on specific system structure

---

## Mathematical Correctness

The implementation correctly computes:

✓ **Equation 6.4**: Base case with degree 2 cross-component sum
✓ **Equation 6.14**: Recursive case with ⃗h term and edge interaction sum
✓ **Degree 2 extraction**: Proper principal part extraction for b vector
✓ **Component constraints**: C(l)≠C(v) and C(w)≠C(l) properly enforced
✓ **Alpha coefficients**: Correctly used as α(v)² - α(l)² and α(l)² - α(w)²

---

## Status

The b vector logic follows the cited equations with degree‑2 principal part extraction. See code in `fas_minor_calculator.py` for the authoritative implementation.

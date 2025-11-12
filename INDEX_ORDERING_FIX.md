# Index Ordering Fix - 2025-11-09

## Issue Identified

Row (0,0,1) was missing the entry c^{0,(0,3)}_{(0,0),(0,3)}u_{0,3}.

## Root Cause

The structure function indices were backwards. The implementation was computing:
```
q_{j,k} = sum_i c^k_{j,i} u_i  (WRONG)
```

But the correct mathematical formula is:
```
q_{j,k} = sum_i c^k_{i,j} u_i  (CORRECT)
```

Where:
- **i** is the summation variable (first index)
- **j** is the row vertex (second index, fixed)

## Key Structure Function: c^k_{i,j}

The structure function c^k_{i,j} represents:
- **k**: edge (superscript)
- **i**: first vertex index (summation variable)
- **j**: second vertex index (row vertex)

## Changes Made

### 1. Documentation Update

Updated constraint comments to use c^k_{i,j} instead of c^k_{j,i}:

```python
"""
Constraints from Tanaka symbol decomposition:
For c^k_{i,j} where k is an edge and i,j are vertices:
1. c^k_{i,j} = 0 when i and j are from different components
2. c^k_{i,j} = 0 when i and j are from the same component, but k is from a different component
3. c^k_{i,j} = 0 when (i,j) or (j,i) is an edge, unless k is that specific edge
4. c^k_{i,j} = 0 when depth(k) > min(depth(i), depth(j))
5. c^k_{i,j} = 0 when depth(i) = depth(j) and (i is root or j is root), unless depth(k) < depth(i)
6. c^k_{i,j} = 1 when (i,j) is an edge and k=(i,j)
7. c^k_{i,j} = -1 when (j,i) is an edge and k=(j,i)
"""
```

### 2. Key Construction in _create_default_structure_functions

**Before (WRONG):**
```python
for g_j, graph_j in enumerate(self.graphs):
    for vertex_j in graph_j.vertices:
        for g_i, graph_i in enumerate(self.graphs):
            for vertex_i in graph_i.vertices:
                key = ('edge', (g_k, edge_k), 'vertex', (g_j, vertex_j), 'vertex', (g_i, vertex_i))
                # This represents c^k_{j,i}
```

**After (CORRECT):**
```python
for g_i, graph_i in enumerate(self.graphs):
    for vertex_i in graph_i.vertices:
        for g_j, graph_j in enumerate(self.graphs):
            for vertex_j in graph_j.vertices:
                key = ('edge', (g_k, edge_k), 'vertex', (g_i, vertex_i), 'vertex', (g_j, vertex_j))
                # This represents c^k_{i,j}
```

Note: Iteration order doesn't matter for generating all combinations, but the key construction order is critical.

### 3. Numeric Constraint Update

**Before (WRONG):**
```python
if edge_ij in graph_k.edges:
    if edge_k == edge_ij:
        self.structure_functions_symbolic[key] = -1  # WRONG sign
elif edge_ji in graph_k.edges:
    if edge_k == edge_ji:
        self.structure_functions_symbolic[key] = 1   # WRONG sign
```

**After (CORRECT):**
```python
if edge_ij in graph_k.edges:
    if edge_k == edge_ij:
        self.structure_functions_symbolic[key] = 1   # CORRECT: c^{(i,j)}_{i,j} = 1
elif edge_ji in graph_k.edges:
    if edge_k == edge_ji:
        self.structure_functions_symbolic[key] = -1  # CORRECT: c^{(j,i)}_{i,j} = -1
```

### 4. Symbol Name Update

**Before:**
```python
symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_j},{vertex_j}),({g_i},{vertex_i})}}'
```

**After:**
```python
symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}'
```

### 5. _compute_q Function

**Before (WRONG):**
```python
# Get structure function c^k_{j,i}
key = (k_type, k_val, j_type, j_val, 'vertex', (g_i, vertex_i))
```

**After (CORRECT):**
```python
# Get structure function c^k_{i,j}
key = (k_type, k_val, 'vertex', (g_i, vertex_i), j_type, j_val)
```

## Verification

### Test Case: Row (0,0,1) for (3,1,5)

**Row vertex:** j = 0
**Layer:** 1 (edges)
**Edges:** {(0,1), (0,2), (0,3), (1,2)}

#### Expected Computation

For edge k = (0,1):
```
q_{0,(0,1)} = sum_i c^{(0,1)}_{i,0} u_i
            = c^{(0,1)}_{1,0} * u_{0,1}  (only i=1 contributes)
            = -1 * u_{0,1}               (because (j,i)=(0,1) is edge, k=(j,i))
            = -u_{0,1}
```

For edge k = (0,2):
```
q_{0,(0,2)} = c^{(0,2)}_{2,0} * u_{0,2} = -u_{0,2}
```

For edge k = (0,3):
```
q_{0,(0,3)} = c^{(0,3)}_{3,0} * u_{0,3} = -u_{0,3}
```

For edge k = (1,2):
```
q_{0,(1,2)} = 0  (no vertex i forms an edge with 0 that matches k=(1,2))
```

#### Actual Output

```
Row (0,0,1) = [-u_{0,1}, -u_{0,2}, -u_{0,3}, 0, 0, 0, 0, 0, 0]
```

**✓ VERIFIED CORRECT**

## Impact on Constraints

All constraints work correctly with the new index ordering:

1. **Component constraint:** i, j, k same component ✓
2. **Depth constraint:** depth(k) ≤ min(depth(i), depth(j)) ✓
3. **Root-depth constraint:** Works correctly ✓
4. **Edge constraint:** Correctly eliminates non-matching edges ✓
5. **Numeric constraint:**
   - c^{(i,j)}_{i,j} = 1 when (i,j) is edge and k=(i,j) ✓
   - c^{(i,j)}_{j,i} = -1 when (i,j) is edge and k=(j,i) ✓

## Statistics (3,1,5)

After the fix:
- Total structure functions: 180
- Zero: 63 (35.0%)
- Numeric ±1: 8 (4.4%)
- Symbolic: 109 (60.6%)

For c^k_{vertex,vertex} only (100 functions):
- Zero: 46%
- Symbolic: 46%
- Numeric ±1: 8%

## Summary

The index ordering fix ensures that:
1. **Row computations are mathematically correct:** q_{j,k} = sum_i c^k_{i,j} u_i
2. **Constraint logic is consistent:** All constraints interpret indices correctly
3. **Numeric values have correct signs:** ±1 values match edge orientation
4. **Missing entries now appear:** Previously missing terms like c^{(0,3)}_{3,0} now correctly appear

**Status:** ✓ FIXED AND VERIFIED

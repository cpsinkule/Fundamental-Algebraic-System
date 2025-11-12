# Depth Implementation

## Date: 2025-11-09

## Summary

Implemented depth computation for vertices and edges in component graphs based on the canonical construction from characteristic tuples.

## Depth Definitions

### 1. Depth of Root Vertices
For a characteristic tuple `(r_0, r_1, ..., r_ω, v)`:
- The first `ω + 1` vertices are **roots** (vertices 0, 1, ..., ω)
- **depth(root_i) = i**

### 2. Depth of Edges
All edges emanate from roots in the canonical construction.
- **depth(edge) = depth(source vertex)**

Since source vertices are always roots:
- Edge starting from root i has depth i

### 3. Depth of Non-Root Vertices
For vertices that are not roots:
- **depth = max(depth of edges containing this vertex)**
- If a non-root vertex appears in no edges: **depth = 0** (default)

## Implementation Details

### ComponentGraph Class

Added three new attributes and methods:

**Attributes:**
- `num_roots`: Number of root vertices (ω + 1)
- `vertex_depths`: Dictionary mapping vertex → depth
- `edge_depths`: Dictionary mapping edge → depth

**Methods:**
- `_compute_vertex_depths()`: Computes depths for all vertices
- `_compute_edge_depths()`: Computes depths for all edges
- `get_vertex_depth(vertex)`: Returns depth of a vertex
- `get_edge_depth(edge)`: Returns depth of an edge

### Modified Constructor

```python
def __init__(self, vertices: List[int], edges: List[Tuple[int, int]], num_roots: int = None):
    """
    Args:
        num_roots: Number of root vertices (for depth computation)
    """
    self.num_roots = num_roots if num_roots is not None else 0
    # ... compute depths automatically
```

### Modified from_characteristic_tuple

```python
@classmethod
def from_characteristic_tuple(cls, char_tuple: Tuple[int, ...]):
    root_degrees = char_tuple[:-1]
    num_roots = len(root_degrees)  # ω + 1
    # ...
    return cls(vertices, edges, num_roots=num_roots)
```

## Examples

### Example 1: (3, 1, 5)

**Characteristic tuple:** (3, 1, 5)
- r_0 = 3, r_1 = 1, v = 5
- ω = 1, so 2 roots (vertices 0, 1)

**Vertices:** 0, 1, 2, 3, 4
**Edges:** (0→1), (0→2), (0→3), (1→2)

**Vertex Depths:**
```
Vertex 0 (root):     depth = 0
Vertex 1 (root):     depth = 1
Vertex 2 (non-root): depth = max(0, 1) = 1  [appears in (0,2) and (1,2)]
Vertex 3 (non-root): depth = 0              [appears in (0,3)]
Vertex 4 (non-root): depth = 0              [no edges]
```

**Edge Depths:**
```
Edge (0,1): depth = 0  [source = root 0]
Edge (0,2): depth = 0  [source = root 0]
Edge (0,3): depth = 0  [source = root 0]
Edge (1,2): depth = 1  [source = root 1]
```

### Example 2: Multi-component System (3,1,5) + (3,1,4)

Each component uses **local vertex numbering**, so both have:
- Roots: 0, 1
- Both components have their own independent depth computations

**Component 0:**
- Vertices 0-4 (local)
- Depths: {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}

**Component 1:**
- Vertices 0-3 (local)
- Depths: {0: 0, 1: 1, 2: 1, 3: 0}

Note: Both components have vertex 0 with depth 0, vertex 1 with depth 1, etc., because they use local numbering.

## Usage

```python
from fas_minor_calculator import ComponentGraph, FASMinorCalculator

# Create graph from characteristic tuple
graph = ComponentGraph.from_characteristic_tuple((3, 1, 5))

# Access depths
v_depth = graph.get_vertex_depth(2)  # Returns 1
e_depth = graph.get_edge_depth((1, 2))  # Returns 1

# For multi-component systems
calc = FASMinorCalculator.from_characteristic_tuples([(3, 1, 5), (3, 1, 4)])
graph0 = calc.graphs[0]
graph1 = calc.graphs[1]

# Each component has its own depths
depth_g0_v2 = graph0.get_vertex_depth(2)  # Component 0, vertex 2
depth_g1_v2 = graph1.get_vertex_depth(2)  # Component 1, vertex 2
```

## Mathematical Correctness

The implementation correctly enforces the depth definitions:

1. ✓ Roots have depth equal to their index
2. ✓ Edges have depth equal to their source vertex (always a root)
3. ✓ Non-root vertices have depth = max depth of edges containing them
4. ✓ Isolated vertices (no edges) have depth 0

## Integration with Local Numbering

The depth system works seamlessly with local vertex numbering:
- Each component has vertices numbered 0, 1, 2, ...
- Each component independently computes depths based on its characteristic tuple
- Depth 0 vertices in different components are distinct: (0, 0) vs (1, 0)
- Depth computations are local to each component

## Next Steps

The depth information can now be used for:
1. Implementing depth-based constraints on structure functions
2. Organizing computations by depth levels
3. Analyzing the graded structure of the fundamental algebraic system
4. Optimizing computations based on depth stratification

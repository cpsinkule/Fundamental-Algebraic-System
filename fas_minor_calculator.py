"""
Fundamental Algebraic System (FAS) Minor Calculator

Symbolic-only calculator for rows of the principal part of the augmented system
[A|b] from a multi-component FAS. It computes entries on-demand according to
the thesis formulas, applying principal-part projections:

- A: degree 1 in vertex variables, with recursion a^{s+1}_{v,w} = h₁(a^s_{v,w})
  (the Σ a·q term is degree 2 and omitted in principal part)
- b: degree 2 in vertex variables, with base/recursive formulas and h₂ extraction

Assumptions and setup:
- Structure functions and α’s are constants for principal part (h acts only on u’s)
- Components are directed graphs with local vertex numbering
- Rows specified by (graph_idx, vertex, layer s≥1)
- Columns are one per edge across all components plus one b column (n-m+1)
"""

from typing import List, Tuple, Dict, Set
import sympy as sp
from sympy import latex
import time


class ComponentGraph:
    """Represents a component graph with vertices and directed edges."""

    def __init__(self, vertices: List[int], edges: List[Tuple[int, int]], num_roots: int = None):
        """
        Initialize a component graph.

        Args:
            vertices: List of vertex labels (integers)
            edges: List of directed edges as tuples (source, target)
            num_roots: Number of root vertices (for depth computation). If None, assume all vertices are non-roots.
        """
        self.vertices = set(vertices)
        self.edges = edges
        self.num_roots = num_roots if num_roots is not None else 0
        self.adjacency = self._build_adjacency()

        # Compute depths for vertices and edges
        self.vertex_depths = self._compute_vertex_depths()
        self.edge_depths = self._compute_edge_depths()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        """Build adjacency list representation."""
        adj = {v: [] for v in self.vertices}
        for src, tgt in self.edges:
            if src in self.vertices and tgt in self.vertices:
                adj[src].append(tgt)
        return adj

    def get_outgoing_edges(self, vertex: int) -> List[int]:
        """Get all vertices that have incoming edges from the given vertex."""
        return self.adjacency.get(vertex, [])

    def _compute_vertex_depths(self) -> Dict[int, int]:
        """
        Compute depths of all vertices.

        Depth rules:
        1. Root vertices (0, 1, ..., num_roots-1): depth(root_i) = i
        2. Non-root vertices: depth = max(depth of edges containing this vertex)
        3. If a non-root vertex appears in no edges: depth = 0

        Returns:
            Dictionary mapping vertex -> depth
        """
        depths = {}

        # 1. Assign depths to roots
        for i in range(self.num_roots):
            if i in self.vertices:
                depths[i] = i

        # 2. Compute edge depths (needed for non-root vertex depths)
        # Edge depth = depth of source vertex
        edge_depths = {}
        for src, tgt in self.edges:
            edge_depths[(src, tgt)] = depths.get(src, 0)

        # 3. Compute non-root vertex depths
        for v in self.vertices:
            if v < self.num_roots:
                continue  # Already assigned

            # Find max depth of edges containing this vertex
            max_depth = 0
            found_edge = False

            for edge in self.edges:
                src, tgt = edge
                if src == v or tgt == v:
                    found_edge = True
                    max_depth = max(max_depth, edge_depths.get(edge, 0))

            # Depth of non-root vertex = max(edge depths) + 1
            depths[v] = max_depth + 1 if found_edge else 0

        return depths

    def _compute_edge_depths(self) -> Dict[Tuple[int, int], int]:
        """
        Compute depths of all edges.

        Depth rule: depth(edge) = depth(source vertex)
        Since all edges emanate from roots in canonical construction,
        depth(edge) = depth of root = index of root.

        Returns:
            Dictionary mapping edge -> depth
        """
        depths = {}

        for src, tgt in self.edges:
            # Depth of edge = depth of source vertex
            depths[(src, tgt)] = self.vertex_depths.get(src, 0)

        return depths

    def get_vertex_depth(self, vertex: int) -> int:
        """Get the depth of a vertex."""
        return self.vertex_depths.get(vertex, 0)

    def get_edge_depth(self, edge: Tuple[int, int]) -> int:
        """Get the depth of an edge."""
        return self.edge_depths.get(edge, 0)

    def __repr__(self):
        return f"Graph(V={sorted(self.vertices)}, E={self.edges})"

    @classmethod
    def from_characteristic_tuple(cls, char_tuple: Tuple[int, ...]):
        """
        Construct a ComponentGraph from a characteristic tuple.

        The characteristic tuple has the form (r_0, r_1, ..., r_omega, v) where:
        - r_0, r_1, ..., r_omega are the out-degrees of root vertices (strictly decreasing)
        - v is the total number of vertices in the component

        Vertices are always numbered locally starting from 0.
        The first omega+1 vertices (0, 1, ..., omega) are roots with depth = index.

        Args:
            char_tuple: Tuple (r_0, r_1, ..., r_omega, v)

        Returns:
            ComponentGraph constructed according to canonical rules

        Example:
            >>> graph = ComponentGraph.from_characteristic_tuple((3, 1, 5))
            >>> # Creates 5 vertices: 0, 1, 2, 3, 4 (local numbering)
            >>> # Roots: 0, 1 (omega = 1, so 2 roots)
            >>> # Edges: (0→1), (0→2), (0→3), (1→2)
            >>> # Depths: vertex 0 (root) = 0, vertex 1 (root) = 1, vertex 2 = 1, vertex 3 = 0, vertex 4 = 0
        """
        # Last element is number of vertices
        v = char_tuple[-1]
        # All preceding elements are root out-degrees
        root_degrees = char_tuple[:-1]
        # Number of roots = omega + 1 = length of root_degrees
        num_roots = len(root_degrees)

        # Create vertices - always start from 0 (local numbering)
        vertices = list(range(v))

        # Create edges according to canonical construction
        # x_i points to x_{i+1}, x_{i+2}, ..., x_{i+r_i}
        edges = []

        for i, r_i in enumerate(root_degrees):
            root_vertex = i
            for j in range(1, r_i + 1):
                target_vertex = root_vertex + j
                edges.append((root_vertex, target_vertex))

        return cls(vertices, edges, num_roots=num_roots)


class FASMinorCalculator:
    """
    Calculator for minors of the fundamental algebraic system.

    The principal part of the system is represented as an augmented matrix [A|b]
    where rows correspond to (vertex, layer) pairs in the component graphs.
    """

    def __init__(self, graphs: List[ComponentGraph], use_symbolic=False,
                 enable_simplification=True, simplification_threshold=10000,
                 use_lazy_structure_functions=True, show_performance_warnings=False):
        """
        Initialize the calculator (symbolic-only).

        Vertices are numbered locally starting from 0 within each component.
        Vertices are referenced by (graph_idx, local_vertex) tuples.

        Args:
            graphs: List of component graphs (local vertex numbering)
            use_symbolic: Must be True (symbolic-only). If False, a ValueError is raised.
            enable_simplification: If True, simplify expressions at higher layers
            simplification_threshold: Skip simplification if expression string exceeds this
            use_lazy_structure_functions: If True, create structure functions on-demand
            show_performance_warnings: If True, print warnings about expression sizes
        """
        self.graphs = graphs
        # Enforce symbolic mode only (numeric minors are not well-defined here)
        if not use_symbolic:
            raise ValueError("Only symbolic mode is supported; numeric computation of minors is not defined.")
        self.use_symbolic = True

        # Performance optimization settings
        self.enable_simplification = enable_simplification
        self.simplification_threshold = simplification_threshold
        self.use_lazy_structure_functions = use_lazy_structure_functions
        self.show_performance_warnings = show_performance_warnings

        # n is the dimension of the manifold = vertices + edges
        self.n = sum(len(g.vertices) + len(g.edges) for g in graphs)
        self.matrix_entries = {}  # Cache for matrix entries

        # Performance optimization caches
        self._h_action_cache = None  # Cache for Hamiltonian vector field action
        self._expanded_expr_cache = {}  # Cache for expanded expressions
        self._q_cache = {}  # Cache for q(j,k) results

        # Performance timing statistics
        self._timing_stats = {
            'build_matrix_entry': 0.0,
            'apply_h': 0.0,
            'extract_principal_part': 0.0,
        }
        self._timing_counts = {key: 0 for key in self._timing_stats.keys()}

        # Create symbolic variables for vertices and edges
        self._initialize_symbolic_variables()

        # Structure functions with flexible indexing: c^a_{b,c}
        # where a, b, c can independently be vertices or edges
        # Key format: (index_type_a, (g_a, a), index_type_b, (g_b, b), index_type_c, (g_c, c))
        # index_type is 'vertex' or 'edge'
        self.structure_functions_symbolic = {}

        # Create structure functions based on lazy loading setting
        if self.use_symbolic and not self.use_lazy_structure_functions:
            self._create_default_structure_functions()

    @classmethod
    def from_characteristic_tuples(cls, char_tuples: List[Tuple[int, ...]], use_symbolic=False,
                                   enable_simplification=True, simplification_threshold=10000,
                                   use_lazy_structure_functions=True, show_performance_warnings=False):
        """
        Create a FASMinorCalculator from characteristic tuples.

        Each characteristic tuple has the form (r_0, r_1, ..., r_omega, v) where:
        - r_0, r_1, ..., r_omega are the out-degrees of root vertices (strictly decreasing)
        - v is the total number of vertices in that component

        Each component uses local vertex numbering starting from 0.
        Vertices are referenced by (graph_idx, local_vertex) tuples.

        Args:
            char_tuples: List of characteristic tuples, one per component
            use_symbolic: Must be True (symbolic-only)
            enable_simplification: If True, simplify expressions at higher layers
            simplification_threshold: Skip simplification if expression string length exceeds this
            use_lazy_structure_functions: If True, create structure functions on-demand
            show_performance_warnings: If True, show warnings about expression sizes

        Returns:
            FASMinorCalculator instance

        Example:
            >>> calc = FASMinorCalculator.from_characteristic_tuples([
            ...     (3, 1, 5),  # Component 0: vertices 0-4 (local), roots with degrees 3, 1
            ...     (3, 1, 4)   # Component 1: vertices 0-3 (local), roots with degrees 3, 1
            ... ])
        """
        graphs = []
        for char_tuple in char_tuples:
            graph = ComponentGraph.from_characteristic_tuple(char_tuple)
            graphs.append(graph)

        return cls(graphs, use_symbolic=use_symbolic,
                  enable_simplification=enable_simplification,
                  simplification_threshold=simplification_threshold,
                  use_lazy_structure_functions=use_lazy_structure_functions,
                  show_performance_warnings=show_performance_warnings)

    def _initialize_symbolic_variables(self):
        """
        Initialize symbolic variables for vertices and edges.

        Creates:
        - u_{g,v} for each vertex v in component g (symbolic variables)
        - u_{g,e} for each edge e in component g (parameters/constants for principal part)

        Vertices are indexed by (graph_idx, local_vertex) tuples.
        Edges are indexed by (graph_idx, local_edge) tuples.
        """
        self.vertex_variables = {}  # Map: (graph_idx, vertex) -> u_{g,v} symbol
        self.edge_variables = {}     # Map: (graph_idx, edge) -> u_{g,e} symbol

        if self.use_symbolic:
            # Create vertex variables (symbolic)
            for g_idx, graph in enumerate(self.graphs):
                for vertex in graph.vertices:
                    key = (g_idx, vertex)
                    self.vertex_variables[key] = sp.Symbol(f'u_{{{g_idx},{vertex}}}')

            # Create edge variables (parameters for principal part)
            for g_idx, graph in enumerate(self.graphs):
                for edge in graph.edges:
                    src, tgt = edge
                    key = (g_idx, edge)
                    self.edge_variables[key] = sp.Symbol(f'u_{{{g_idx},({src},{tgt})}}')

            # Create set of vertex symbols for O(1) lookup in _get_vertex_degree()
            self._vertex_symbol_set = set(self.vertex_variables.values())
        else:
            self._vertex_symbol_set = set()

    def _get_structure_function(self, key: Tuple) -> any:
        """
        Get a structure function value, creating it on-demand if using lazy loading.

        Args:
            key: Structure function key tuple

        Returns:
            Structure function value
        """
        # If already cached, return it
        if key in self.structure_functions_symbolic:
            return self.structure_functions_symbolic[key]

        # If not using lazy loading, return 0 (function wasn't created)
        if not self.use_lazy_structure_functions:
            return 0

        # Lazy create the structure function
        return self._create_structure_function(key)

    def _create_structure_function(self, key: Tuple) -> any:
        """
        Create a single structure function based on the constraints.

        Args:
            key: Structure function key tuple

        Returns:
            Structure function value (symbol, 0, 1, or -1)
        """
        # Parse the key
        index_type_a, val_a, index_type_b, val_b, index_type_c, val_c = key

        # Handle different cases based on index types
        if index_type_a == 'edge' and index_type_b == 'vertex' and index_type_c == 'vertex':
            # c^k_{i,j} where k is edge, i and j are vertices
            g_k, edge_k = val_a
            g_i, vertex_i = val_b
            g_j, vertex_j = val_c
            k_src, k_tgt = edge_k

            graph_k = self.graphs[g_k]

            # Constraint 0: c^k_{i,j} = 0 when i = j
            if g_i == g_j and vertex_i == vertex_j:
                self.structure_functions_symbolic[key] = 0
                return 0

            # Constraint 1: c^k_{i,j} = 0 unless i, j, k are all from the same component
            if g_i != g_j or g_i != g_k:
                self.structure_functions_symbolic[key] = 0
                return 0

            # All from same component: g_i = g_j = g_k
            # Constraint 2: Depth constraint - depth(k) > min(depth(i), depth(j))
            depth_k = graph_k.get_edge_depth(edge_k)
            depth_i = graph_k.get_vertex_depth(vertex_i)
            depth_j = graph_k.get_vertex_depth(vertex_j)
            min_depth_ij = min(depth_i, depth_j)

            if depth_k > min_depth_ij:
                self.structure_functions_symbolic[key] = 0
                return 0

            # Constraint 3: Root-depth constraint
            if depth_i == depth_j:
                i_is_root = vertex_i < graph_k.num_roots
                j_is_root = vertex_j < graph_k.num_roots

                if i_is_root or j_is_root:
                    if depth_k >= depth_i:
                        self.structure_functions_symbolic[key] = 0
                        return 0

            # Constraint 4: Edge constraints
            edge_ij = (vertex_i, vertex_j)
            edge_ji = (vertex_j, vertex_i)

            if edge_ij in graph_k.edges:
                if edge_k == edge_ij:
                    self.structure_functions_symbolic[key] = 1
                    return 1
                else:
                    self.structure_functions_symbolic[key] = 0
                    return 0
            elif edge_ji in graph_k.edges:
                if edge_k == edge_ji:
                    self.structure_functions_symbolic[key] = -1
                    return -1
                else:
                    self.structure_functions_symbolic[key] = 0
                    return 0
            else:
                # Create symbolic variable
                symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}'
                result = sp.Symbol(symbol_name)
                self.structure_functions_symbolic[key] = result
                return result

        elif index_type_a == 'edge' and index_type_b == 'edge' and index_type_c == 'vertex':
            # c^k_{l,i} where k and l are edges, i is vertex
            g_k, edge_k = val_a
            g_l, edge_l = val_b
            g_i, vertex_i = val_c
            k_src, k_tgt = edge_k
            l_src, l_tgt = edge_l

            # Create symbolic variable for all combinations
            symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}'
            result = sp.Symbol(symbol_name)
            self.structure_functions_symbolic[key] = result
            return result

        elif index_type_a == 'vertex' and index_type_b == 'vertex' and index_type_c == 'vertex':
            # c^l_{w,v} where l, w, v are all vertices
            g_l, vertex_l = val_a
            g_w, vertex_w = val_b
            g_v, vertex_v = val_c

            # Constraint 0: Diagonal zero
            if g_w == g_v and vertex_w == vertex_v:
                self.structure_functions_symbolic[key] = 0
                return 0

            # Constraint 1: Edge constraint
            if g_w == g_v:
                graph_for_check = self.graphs[g_w]
                edge_wv = (vertex_w, vertex_v)
                edge_vw = (vertex_v, vertex_w)

                if edge_wv in graph_for_check.edges or edge_vw in graph_for_check.edges:
                    self.structure_functions_symbolic[key] = 0
                    return 0

            # Constraint 2: Component locality constraint
            # If w and v are in the same component, then l must also be in that component
            if g_w == g_v and g_l != g_w:
                self.structure_functions_symbolic[key] = 0
                return 0

            # Create symbolic structure function
            symbol_name = f'c^{{({g_l},{vertex_l})}}_{{({g_w},{vertex_w}),({g_v},{vertex_v})}}'
            result = sp.Symbol(symbol_name)
            self.structure_functions_symbolic[key] = result
            return result

        else:
            # Unknown key format, return 0
            self.structure_functions_symbolic[key] = 0
            return 0

    def _create_default_structure_functions(self):
        """
        Create default symbolic structure functions c^a_{b,c}.

        With local vertex numbering, vertices and edges are referenced by
        (graph_idx, local_index) tuples. The key format is:
        ('vertex'|'edge', (g,a), 'vertex'|'edge', (g,b), 'vertex'|'edge', (g,c))

        Constraints from Tanaka symbol decomposition:
        For c^k_{i,j} where k is an edge and i,j are vertices:
        1. c^k_{i,j} = 0 when i and j are from different components
        2. c^k_{i,j} = 0 when i and j are from the same component, but k is from a different component
        3. c^k_{i,j} = 0 when (i,j) or (j,i) is an edge, unless k is that specific edge
        4. c^k_{i,j} = 0 when depth(k) > min(depth(i), depth(j))
        5. c^k_{i,j} = 0 when depth(i) = depth(j) and (i is root or j is root), unless depth(k) < depth(i)
        6. c^k_{i,j} = 1 when (i,j) is an edge and k=(i,j)
        7. c^k_{i,j} = -1 when (j,i) is an edge and k=(j,i)

        Therefore: c^k_{i,j} is non-zero only when:
        - i, j, and k are all from the SAME component, AND
        - if i and j form an edge, then k must be exactly that edge, AND
        - depth(k) ≤ min(depth(i), depth(j)), AND
        - if depth(i) = depth(j) and at least one is a root, then depth(k) < depth(i)
        """
        # For each component, create structure functions for its edges and vertices
        for g_k, graph_k in enumerate(self.graphs):
            for edge_k in graph_k.edges:
                k_src, k_tgt = edge_k

                # c^k_{i,j} - needed for q_{j,k} = sum_i c^k_{i,j} u_i
                # Sum over ALL vertices in ALL components (most will be zero due to constraints)
                for g_i, graph_i in enumerate(self.graphs):
                    for vertex_i in graph_i.vertices:

                        for g_j, graph_j in enumerate(self.graphs):
                            for vertex_j in graph_j.vertices:

                                key = ('edge', (g_k, edge_k), 'vertex', (g_i, vertex_i), 'vertex', (g_j, vertex_j))

                                # Constraint 0: c^k_{i,j} = 0 when i = j
                                if g_i == g_j and vertex_i == vertex_j:
                                    self.structure_functions_symbolic[key] = 0
                                    continue

                                # Constraint 1: c^k_{i,j} = 0 unless i, j, k are all from the same component
                                if g_i != g_j or g_i != g_k:
                                    self.structure_functions_symbolic[key] = 0
                                    continue

                                # All from same component: g_i = g_j = g_k
                                # Constraint 2: Depth constraint - depth(k) > min(depth(i), depth(j))
                                depth_k = graph_k.get_edge_depth(edge_k)
                                depth_i = graph_k.get_vertex_depth(vertex_i)
                                depth_j = graph_k.get_vertex_depth(vertex_j)
                                min_depth_ij = min(depth_i, depth_j)

                                if depth_k > min_depth_ij:
                                    self.structure_functions_symbolic[key] = 0
                                    continue

                                # Constraint 3: Root-depth constraint
                                # If depth(i) = depth(j) and (i is root or j is root), then c^k_{j,i} = 0 unless depth(k) < depth(i)
                                if depth_i == depth_j:
                                    # Check if i or j is a root (roots are vertices 0, 1, ..., num_roots-1)
                                    i_is_root = vertex_i < graph_k.num_roots
                                    j_is_root = vertex_j < graph_k.num_roots

                                    if i_is_root or j_is_root:
                                        # Constraint applies: require depth(k) < depth(i) = depth(j)
                                        if depth_k >= depth_i:
                                            self.structure_functions_symbolic[key] = 0
                                            continue

                                # Constraint 4: If (i,j) or (j,i) form an edge, then c^k_{i,j} = 0 unless k is that edge
                                # Constraint 5: If (i,j) is edge and k=(i,j), then c^k_{i,j} = 1
                                #               If (j,i) is edge and k=(j,i), then c^k_{i,j} = -1
                                edge_ij = (vertex_i, vertex_j)
                                edge_ji = (vertex_j, vertex_i)

                                if edge_ij in graph_k.edges:
                                    # (i,j) is an edge, so k must be (i,j)
                                    if edge_k == edge_ij:
                                        # c^k_{i,j} = 1 when k = (i,j)
                                        self.structure_functions_symbolic[key] = 1
                                    else:
                                        self.structure_functions_symbolic[key] = 0
                                elif edge_ji in graph_k.edges:
                                    # (j,i) is an edge, so k must be (j,i)
                                    if edge_k == edge_ji:
                                        # c^k_{i,j} = -1 when k = (j,i)
                                        self.structure_functions_symbolic[key] = -1
                                    else:
                                        self.structure_functions_symbolic[key] = 0
                                else:
                                    # i and j don't form an edge, so this constraint doesn't apply
                                    symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}'
                                    self.structure_functions_symbolic[key] = sp.Symbol(symbol_name)

                # c^k_{edge,vertex} - needed for q_{lw} where l is an edge
                for g_l, graph_l in enumerate(self.graphs):
                    for edge_l in graph_l.edges:
                        l_src, l_tgt = edge_l

                        for g_i, graph_i in enumerate(self.graphs):
                            for vertex_i in graph_i.vertices:

                                key = ('edge', (g_k, edge_k), 'edge', (g_l, edge_l), 'vertex', (g_i, vertex_i))

                                # No constraints for Type 2 structure functions (beyond universal diagonal zero)
                                # Create symbolic variable for all combinations
                                symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}'
                                self.structure_functions_symbolic[key] = sp.Symbol(symbol_name)

        # Create structure functions with VERTEX superscripts (needed for b vector equation 6.4)
        # These are c^l_{w,v} where l, w, v are all vertices
        # Key constraint: These are specifically for CROSS-COMPONENT interactions (C(l) ≠ C(v))
        for g_l, graph_l in enumerate(self.graphs):
            for vertex_l in graph_l.vertices:
                # l is the superscript vertex

                # Create c^l_{w,v} for all vertices w and v
                for g_w, graph_w in enumerate(self.graphs):
                    for vertex_w in graph_w.vertices:

                        for g_v, graph_v in enumerate(self.graphs):
                            for vertex_v in graph_v.vertices:

                                key = ('vertex', (g_l, vertex_l), 'vertex', (g_w, vertex_w), 'vertex', (g_v, vertex_v))

                                # Constraint 0: Diagonal zero - c^l_{w,v} = 0 when w = v
                                if g_w == g_v and vertex_w == vertex_v:
                                    self.structure_functions_symbolic[key] = 0
                                    continue

                                # Constraint 1: Edge constraint - c^l_{w,v} = 0 when (w,v) or (v,w) is an edge
                                if g_w == g_v:  # Can only form edge if in same component
                                    graph_for_check = self.graphs[g_w]
                                    edge_wv = (vertex_w, vertex_v)
                                    edge_vw = (vertex_v, vertex_w)

                                    if edge_wv in graph_for_check.edges or edge_vw in graph_for_check.edges:
                                        self.structure_functions_symbolic[key] = 0
                                        continue

                                # Constraint 2: Component locality constraint (match lazy creation)
                                # If w and v are in the same component, then l must also be in that component
                                if g_w == g_v and g_l != g_w:
                                    self.structure_functions_symbolic[key] = 0
                                    continue

                                # Create symbolic structure function
                                # All other combinations allowed (cross-component, etc.)
                                symbol_name = f'c^{{({g_l},{vertex_l})}}_{{({g_w},{vertex_w}),({g_v},{vertex_v})}}'
                                self.structure_functions_symbolic[key] = sp.Symbol(symbol_name)

    def set_structure_functions(self, functions: Dict[Tuple[int, int, int], any]):
        """
        Set structure function values.

        Args:
            functions: Dictionary mapping (graph_idx, vertex, layer) to function value
        """
        self.structure_functions = functions

    def set_alpha_values(self, alphas: Dict[int, any]):
        """
        Set alpha coefficient values.

        Alpha coefficients are component-based: all vertices and edges
        within the same component share the same alpha value.

        Args:
            alphas: Dictionary mapping graph_idx to alpha value
                   Example: {0: alpha_0, 1: alpha_1}
        """
        self.alphas = alphas

    def set_b_values(self, b_values: Dict[Tuple[int, int, int], any]):
        """
        Set b vector values.

        Args:
            b_values: Dictionary mapping (graph_idx, vertex, layer) to b value
        """
        self.b_values = b_values

    def get_vertex_component_map(self) -> Dict[int, int]:
        """
        Get a mapping from vertex labels to component indices.

        Returns:
            Dictionary mapping vertex label -> component index

        Example:
            >>> calc = FASMinorCalculator.from_characteristic_tuples([(3, 1, 5), (3, 1, 4)])
            >>> mapping = calc.get_vertex_component_map()
            >>> # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1}
        """
        mapping = {}
        for g_idx, graph in enumerate(self.graphs):
            for vertex in graph.vertices:
                mapping[vertex] = g_idx
        return mapping

    # NOTE: The following methods are deprecated with local vertex numbering
    # Vertices are no longer globally unique; they are referenced by (graph_idx, local_vertex) tuples
    # These methods are kept for backward compatibility but should not be used with local numbering

    # def _find_vertex_graph(self, vertex: int) -> int:
    #     """DEPRECATED: Not applicable with local vertex numbering."""
    #     raise NotImplementedError("_find_vertex_graph is not applicable with local vertex numbering. "
    #                               "Use (graph_idx, vertex) tuples instead.")

    # def _find_edge_graph(self, edge: Tuple[int, int]) -> int:
    #     """DEPRECATED: Not applicable with local vertex numbering."""
    #     raise NotImplementedError("_find_edge_graph is not applicable with local vertex numbering. "
    #                               "Use (graph_idx, edge) tuples instead.")

    def _get_edge_by_index(self, column_index: int) -> Tuple[int, Tuple[int, int]]:
        """
        Get the edge corresponding to a column index.

        Args:
            column_index: 0-based column index (0 to total_edges-1)

        Returns:
            Tuple of (graph_idx, edge) where edge is (src, tgt)

        Raises:
            ValueError: If column_index is out of range
        """
        current_idx = 0
        for g_idx, graph in enumerate(self.graphs):
            if current_idx + len(graph.edges) > column_index:
                edge_idx = column_index - current_idx
                return (g_idx, graph.edges[edge_idx])
            current_idx += len(graph.edges)
        raise ValueError(f"Column index {column_index} out of range (total edges: {current_idx})")

    # NOTE: The following methods are deprecated with local vertex numbering
    # Use get_row(graph_idx, vertex, layer) instead

    # def get_A_entry(self, s: int, v: int, c: int) -> any:
    #     """DEPRECATED: Not applicable with local vertex numbering. Use get_row() instead."""
    #     raise NotImplementedError("get_A_entry is deprecated. Use get_row(graph_idx, vertex, layer) instead.")

    # def get_b_entry(self, s: int, v: int) -> any:
    #     """DEPRECATED: Not applicable with local vertex numbering. Use get_row() instead."""
    #     raise NotImplementedError("get_b_entry is deprecated. Use get_row(graph_idx, vertex, layer) instead.")

    def get_row(self, graph_idx: int, vertex: int, layer: int) -> any:
        """
        Get a complete row of the augmented system [A|b].

        Returns all entries for the specified (graph_idx, vertex, layer) row,
        including one entry per edge plus the b column.

        Args:
            graph_idx: Component graph index
            vertex: Vertex label
            layer: Layer number (>= 1)

        Returns:
            Row vector with (total_edges + 1) entries as a SymPy Matrix (1 × N)

        Raises:
            ValueError: If layer < 1 or graph_idx/vertex invalid

        Example:
            >>> calc = FASMinorCalculator.from_characteristic_tuples([(3, 1, 5), (3, 1, 4)])
            >>> # Get complete row for component 0, vertex 0, layer 1
            >>> row = calc.get_row(0, 0, 1)
            >>> # row contains [a^1_{0,e1}, a^1_{0,e2}, ..., b^1_0]
        """
        if layer < 1:
            raise ValueError(f"Layer must be >= 1 (got {layer})")

        if graph_idx < 0 or graph_idx >= len(self.graphs):
            raise ValueError(f"Invalid graph_idx {graph_idx} (must be 0-{len(self.graphs)-1})")

        graph = self.graphs[graph_idx]
        if vertex not in graph.vertices:
            raise ValueError(f"Vertex {vertex} not in graph {graph_idx}")

        # Build row specification
        row_spec = (graph_idx, vertex, layer)

        # Generate all column specifications
        col_specs = []

        # Add one column for each edge across all graphs
        for g_idx, g in enumerate(self.graphs):
            for edge in g.edges:
                col_specs.append(('edge', g_idx, edge))

        # Add b column
        col_specs.append(('b', None, None))

        # Compute each entry in the row (symbolic only)
        num_cols = len(col_specs)
        row = sp.Matrix.zeros(1, num_cols)
        for j, col_spec in enumerate(col_specs):
            row[j] = self.build_matrix_entry(row_spec, col_spec)
        return row

    # def get_row_by_vertex(self, vertex: int, layer: int) -> any:
    #     """DEPRECATED: Not applicable with local vertex numbering. Use get_row(graph_idx, vertex, layer) instead."""
    #     raise NotImplementedError("get_row_by_vertex is deprecated. Use get_row(graph_idx, vertex, layer) instead.")

    def _compute_q(self, j, k, graph_idx: int) -> any:
        """
        Compute q_{j,k} per Eq. 5.12: q_{j,k} = Σ_{i∈V} c^k_{i,j} u_i.

        Principal-part conventions:
        - Sum over all vertices i in all components
        - For k, only edge superscripts are considered (edge variables are constants)
        - When j is a vertex: use Type 1 c^k_{i,j}
        - When j is an edge:   use Type 2 c^k_{j,i} (implemented as c^k_{l,i})

        Args:
            j: Index (g_j, local_j) where local_j can be vertex or edge
            k: Edge index (g_k, local_edge)
            graph_idx: Component graph index (kept for compatibility, not used)

        Returns:
            Symbolic expression (sum over all vertices)
        """
        if not self.use_symbolic:
            raise ValueError("q_{j,k} is only defined symbolically in this calculator.")

        # Symbolic mode: q_{j,k} = Σ_{i∈V} c^k_{i,j} u_i
        # Add a cache as q is queried frequently with same (j,k)
        g_j, local_j = j
        j_type = 'vertex' if isinstance(local_j, int) else 'edge'
        g_k, local_k = k
        cache_key = (j_type, (g_j, local_j), (g_k, local_k))
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        result = 0

        # Determine the type of j
        if isinstance(local_j, int):
            j_type = 'vertex'
            j_val = (g_j, local_j)
        else:
            j_type = 'edge'
            j_val = (g_j, local_j)

        # k must be an edge
        k_type = 'edge'
        k_val = (g_k, local_k)

        # Sum over all vertices in ALL components
        for g_i, graph_i in enumerate(self.graphs):
            for vertex_i in graph_i.vertices:
                # Get structure function with correct index ordering
                # If j is a vertex: use c^k_{i,j}  -> ('edge', k), ('vertex', i), ('vertex', j)
                # If j is an edge:  use c^k_{l,i}  -> ('edge', k), ('edge', l=j), ('vertex', i)
                if j_type == 'vertex':
                    key = (k_type, k_val, 'vertex', (g_i, vertex_i), 'vertex', j_val)
                else:  # j is an edge
                    key = (k_type, k_val, 'edge', j_val, 'vertex', (g_i, vertex_i))
                c_coeff = self._get_structure_function(key)

                # Get vertex variable u_i
                u_i = self.vertex_variables.get((g_i, vertex_i), 0)

                # Add to sum
                if c_coeff != 0 and u_i != 0:
                    result = result + c_coeff * u_i

        # Cache and return
        self._q_cache[cache_key] = result
        return result

    def _build_h_action(self) -> Dict:
        """
        Build the action of the Hamiltonian vector field ⃗h on all variables.

        This computation is expensive but depends only on the graph structure
        and structure functions, not on the specific expression being operated on.
        Therefore it is cached for reuse across multiple _apply_h calls.

        Returns:
            Dictionary mapping each variable to its image under ⃗h
        """
        h_action = {}

        # For each vertex variable u_{g_j,j}
        for g_j, graph_j in enumerate(self.graphs):
            for vertex_j in graph_j.vertices:
                u_j = self.vertex_variables[(g_j, vertex_j)]
                # ⃗h(u_j) = Σ_{k∈E(C(j))} Σ_{i∈V(C(j))} c^k_{i,j} u_i u_k
                # Principal part: only same-component contributions are nonzero
                result = 0

                # Sum over edges and vertices in the same component as j
                for edge_k in graph_j.edges:
                    u_k = self.edge_variables[(g_j, edge_k)]
                    for vertex_i in graph_j.vertices:
                        u_i = self.vertex_variables[(g_j, vertex_i)]
                        # Get structure function c^k_{i,j}
                        key = ('edge', (g_j, edge_k), 'vertex', (g_j, vertex_i), 'vertex', (g_j, vertex_j))
                        c_coeff = self._get_structure_function(key)
                        if c_coeff != 0:
                            result = result + c_coeff * u_i * u_k
                h_action[u_j] = result

        # For edge variables, ⃗h acts as 0 (they're constants for principal part)
        for g_e, graph_e in enumerate(self.graphs):
            for edge in graph_e.edges:
                u_e = self.edge_variables[(g_e, edge)]
                h_action[u_e] = 0

        return h_action

    def _apply_h(self, expr, graph_idx: int) -> any:
        """
        Apply the Hamiltonian vector field ⃗h to an expression.

        For a vertex variable u_j:
        ⃗h(u_j) = Σ_{k∈E(C(j))} Σ_{i∈V(C(j))} c^k_{i,j} u_i u_k
        (edges and vertices in the same component as j).

        For more complex expressions, apply the derivation rule.

        Args:
            expr: Symbolic expression
            graph_idx: Component graph index (unused, kept for compatibility)

        Returns:
            Result of applying ⃗h to the expression
        """
        if not self.use_symbolic:
            # Symbolic-only calculator
            return 0

        # If expr is 0 or a constant, return 0
        if expr == 0 or not expr.free_symbols:
            return 0

        # Use cached h_action if available, otherwise build it
        if self._h_action_cache is None:
            self._h_action_cache = self._build_h_action()

        # Apply ⃗h to the expression using substitution and expansion
        # ⃗h is a derivation, so ⃗h(f*g) = ⃗h(f)*g + f*⃗h(g)
        result = self._apply_derivation(expr, self._h_action_cache)
        return result

    def _apply_derivation(self, expr, action_map: Dict) -> any:
        """
        Apply a derivation to an expression given how it acts on generators.

        Args:
            expr: Symbolic expression
            action_map: Dictionary mapping symbols to their images under the derivation

        Returns:
            Result of applying the derivation
        """
        if not self.use_symbolic:
            return 0

        # If expr is a sum, apply linearly
        if expr.is_Add:
            return sum(self._apply_derivation(arg, action_map) for arg in expr.args)

        # If expr is a product, use Leibniz rule
        if expr.is_Mul:
            result = 0
            args = list(expr.args)
            for i in range(len(args)):
                # Apply derivation to i-th factor, keep others unchanged
                term = 1
                for j in range(len(args)):
                    if i == j:
                        term = term * self._apply_derivation(args[j], action_map)
                    else:
                        term = term * args[j]
                result = result + term
            return result

        # If expr is a power, apply chain rule with exponent treated as constant
        # h(base**exp) = (d/d base)(base**exp) * h(base) when exp is constant w.r.t. derivation
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            base_h = self._apply_derivation(base, action_map)
            if base_h == 0:
                return 0
            # If exponent is a number (including Integer), use explicit formula; otherwise use sympy diff
            try:
                if exp.is_Number:
                    return exp * (base ** (exp - 1)) * base_h
                else:
                    return sp.diff(base ** exp, base) * base_h
            except Exception:
                # Fallback: no contribution if differentiation fails
                return 0

        # If expr is a symbol, look it up in action_map
        if expr.is_Symbol:
            return action_map.get(expr, 0)

        # If expr is a number or constant, derivation gives 0
        return 0

    def _extract_principal_part(self, expr, target_vertex_degree: int) -> any:
        """
        Extract the part of an expression with specified vertex degree.

        For the principal part of A matrix, we want vertex degree 1.
        For the principal part of b vector, we allow vertex degree up to 2.

        Args:
            expr: Symbolic expression
            target_vertex_degree: Desired vertex degree (1 for A, up to 2 for b)

        Returns:
            Part of expression with the specified vertex degree
        """
        if not self.use_symbolic or expr == 0:
            return expr

        # Try to get expanded expression from cache
        # Use hash of expression as cache key
        expr_hash = hash(expr)
        if expr_hash in self._expanded_expr_cache:
            expanded = self._expanded_expr_cache[expr_hash]
        else:
            # Expand the expression and cache it
            expanded = sp.expand(expr)
            # Only cache if not too many entries (prevent unbounded memory growth)
            if len(self._expanded_expr_cache) < 1000:
                self._expanded_expr_cache[expr_hash] = expanded

        # Extract terms with target vertex degree
        if expanded.is_Add:
            result = 0
            for term in expanded.args:
                if self._get_vertex_degree(term) == target_vertex_degree:
                    result = result + term
            return result
        else:
            if self._get_vertex_degree(expanded) == target_vertex_degree:
                return expanded
            return 0

    def _get_vertex_degree(self, expr) -> int:
        """
        Get the total degree in vertex variables for an expression term.

        Args:
            expr: Symbolic expression (should be a single term)

        Returns:
            Total degree in vertex variables
        """
        if not self.use_symbolic or expr == 0:
            return 0

        degree = 0
        # Get all symbols in the expression
        for symbol in expr.free_symbols:
            # Check if this symbol is a vertex variable (O(1) lookup using set)
            if symbol in self._vertex_symbol_set:
                # Count the power of this symbol in the expression
                degree += expr.as_coeff_exponent(symbol)[1]

        return degree

    def _smart_simplify(self, expr) -> any:
        """
        Intelligently simplify an expression based on size and settings.

        This method checks expression size and settings before applying
        simplification, avoiding expensive operations when unnecessary.

        Args:
            expr: Symbolic expression to simplify

        Returns:
            Simplified expression (or original if simplification is skipped)
        """
        if not self.use_symbolic or expr == 0:
            return expr

        # Check if simplification is enabled
        if not self.enable_simplification:
            return expr

        # Check expression size
        expr_str = str(expr)
        expr_size = len(expr_str)

        # Show warning if expression is large
        if self.show_performance_warnings and expr_size > self.simplification_threshold:
            print(f"Warning: Expression size ({expr_size}) exceeds threshold ({self.simplification_threshold}). Skipping simplification.")

        # Skip simplification if expression is too large
        if expr_size > self.simplification_threshold:
            return expr

        # Use faster simplification methods
        # sp.cancel() is much faster than sp.simplify() for rational expressions
        try:
            return sp.cancel(expr)
        except:
            # If cancel fails, return original expression
            return expr

    def build_matrix_entry(self, row_spec: Tuple[int, int, int],
                          col_spec: Tuple) -> any:
        """
        Build a single matrix entry for the principal part on-demand (symbolic).

        This method computes individual entries without building the entire
        infinite system, only calculating what's needed for the requested minor.

        Implementation follows Sinkule thesis Proposition 5.4 (equations 5.16-5.17).

        Recursive formula for A matrix (eq 5.16):
          a¹_{v,w} = q_{vw}                                                    (layer 1: base case)
          a^{s+1}_{v,w} = ⃗h₁(a^s_{v,w}) + Σ_{l∈E(Γ)} a^s_{v,l} q_{lw}      (layer s+1: recursive)

        where:
          - q_{jk} = Σ_{v∈V} c^k_{j,v} u_v  (equation 5.12)
          - ⃗h(u_j) = Σ_{k∈E} Σ_{i∈V} c^k_{i,j} u_i u_k
          - ⃗h₁ means extract the vertex degree 1 part after applying ⃗h

        Args:
            row_spec: (graph_idx, vertex, layer) for the row
            col_spec: Either ('edge', graph_idx, (src, tgt)) for edge columns
                     or ('b', None, None) for the b vector column

        Returns:
            Symbolic expression
        """
        row_graph, row_vertex, row_layer = row_spec

        # Check if this is the b column
        if col_spec[0] == 'b':
            # b column computation using recursive formulas from Sinkule thesis
            # Equation 6.4 (layer 1 base case) and Equation 6.14 (recursive case)
            # CRITICAL: Principal part of b has degree 2 in vertex variables, not degree 1

            # Check cache first
            cache_key_b = (row_graph, row_vertex, row_layer, 'b')
            if cache_key_b in self.matrix_entries:
                return self.matrix_entries[cache_key_b]

            if not self.use_symbolic:
                raise ValueError("b-vector entries are only defined symbolically in this calculator.")

            # Layer 1: Base case (equation 6.4)
            # b¹_v = Σ_{l,w ∈ V(Γ), C(l)≠C(v)} (α(v)² - α(l)²) c^l_{w,v} u_w u_l
            if row_layer == 1:
                result = 0

                # Get alpha for vertex v (component-based)
                alpha_v_key = row_graph
                if hasattr(self, 'alphas') and alpha_v_key in self.alphas:
                    alpha_v = self.alphas[alpha_v_key]
                else:
                    # Use Unicode Greek alpha for better default display
                    alpha_v = sp.Symbol(f'α_{{{row_graph}}}')

                # Sum over all vertices l in different components
                for g_l, graph_l in enumerate(self.graphs):
                    # Constraint: C(l) ≠ C(v), so skip same component
                    if g_l == row_graph:
                        continue

                    for vertex_l in graph_l.vertices:
                        # Get alpha for vertex l (component-based)
                        alpha_l_key = g_l
                        if hasattr(self, 'alphas') and alpha_l_key in self.alphas:
                            alpha_l = self.alphas[alpha_l_key]
                        else:
                            alpha_l = sp.Symbol(f'α_{{{g_l}}}')

                        # Get u_l (vertex variable)
                        u_l = self.vertex_variables.get((g_l, vertex_l), 0)

                        # Sum over all vertices w (all components)
                        for g_w, graph_w in enumerate(self.graphs):
                            for vertex_w in graph_w.vertices:
                                # Get structure function c^l_{w,v}
                                # Key format: ('vertex', l, 'vertex', w, 'vertex', v)
                                key = ('vertex', (g_l, vertex_l), 'vertex', (g_w, vertex_w), 'vertex', (row_graph, row_vertex))
                                c_coeff = self._get_structure_function(key)

                                if c_coeff != 0 and u_l != 0:
                                    # Get u_w (vertex variable)
                                    u_w = self.vertex_variables.get((g_w, vertex_w), 0)

                                    if u_w != 0:
                                        # Add term: (α(v)² - α(l)²) c^l_{w,v} u_w u_l
                                        term = (alpha_v**2 - alpha_l**2) * c_coeff * u_w * u_l
                                        result = result + term

                # Cache and return (already degree 2 by construction)
                self.matrix_entries[cache_key_b] = result
                return result

            # Layer s > 1: Recursive case (equation 6.14)
            # b^{s+1}_v = [⃗h(b^s_v)]_(2) + Σ_{l,w ∈ E(Γ), C(l)=C(v), C(w)≠C(l)} (α(l)² - α(w)²) a^s_{v,l} q_{l,w} u_w

            # Get b^s_v (previous layer)
            prev_layer_spec = (row_graph, row_vertex, row_layer - 1)
            col_spec_b = ('b', None, None)
            b_prev = self.build_matrix_entry(prev_layer_spec, col_spec_b)

            # Compute ⃗h(b^s_v) and extract degree 2 part
            h_term = self._apply_h(b_prev, row_graph)
            # CRITICAL: Extract degree 2, not degree 1!
            h2_term = self._extract_principal_part(h_term, 2)

            # Compute sum term: Σ_{l,w ∈ E(Γ), C(l)=C(v), C(w)≠C(l)} (α(l)² - α(w)²) a^s_{v,l} q_{l,w} u_w
            sum_term = 0

            # Sum over edges l in same component as v
            graph_v = self.graphs[row_graph]
            for edge_l in graph_v.edges:
                l_src, l_tgt = edge_l

                # Get a^s_{v,l} (from previous layer)
                col_spec_l = ('edge', row_graph, edge_l)
                a_s_vl = self.build_matrix_entry(prev_layer_spec, col_spec_l)

                # Get alpha for edge l (component-based)
                alpha_l_key = row_graph
                if hasattr(self, 'alphas') and alpha_l_key in self.alphas:
                    alpha_l = self.alphas[alpha_l_key]
                else:
                    alpha_l = sp.Symbol(f'α_{{{row_graph}}}')

                # Sum over edges w in different components
                for g_w, graph_w in enumerate(self.graphs):
                    # Constraint: C(w) ≠ C(l), so skip same component
                    if g_w == row_graph:
                        continue

                    for edge_w in graph_w.edges:
                        w_src, w_tgt = edge_w

                        # Get alpha for edge w (component-based)
                        alpha_w_key = g_w
                        if hasattr(self, 'alphas') and alpha_w_key in self.alphas:
                            alpha_w = self.alphas[alpha_w_key]
                        else:
                            alpha_w = sp.Symbol(f'α_{{{g_w}}}')

                        # Get q_{l,w} where l and w are both edges
                        q_lw = self._compute_q((row_graph, edge_l), (g_w, edge_w), row_graph)

                        # Get u_w (edge variable)
                        u_w = self.edge_variables.get((g_w, edge_w), 0)

                        # Add term: (α(l)² - α(w)²) a^s_{v,l} q_{l,w} u_w
                        if a_s_vl != 0 and q_lw != 0 and u_w != 0:
                            term = (alpha_l**2 - alpha_w**2) * a_s_vl * q_lw * u_w
                            sum_term = sum_term + term

            # Combine both terms
            result = h2_term + sum_term

            # Simplify for deeper layers to prevent expression growth
            if row_layer > 2:
                result = self._smart_simplify(result)

            # Extract degree 2 principal part
            result = self._extract_principal_part(result, 2)

            # Cache and return
            self.matrix_entries[cache_key_b] = result
            return result

        # Otherwise it's an edge column
        col_type, col_graph, edge_w = col_spec

        # Only same-graph contributions in principal part
        if row_graph != col_graph:
            return 0

        # Check cache first
        cache_key = (row_graph, row_vertex, row_layer, edge_w)
        if cache_key in self.matrix_entries:
            return self.matrix_entries[cache_key]

        # Layer 1: Base case (equation 5.16, first line)
        # a¹_{v,w} = q_{vw} = Σ_{v'∈V} c^w_{v,v'} u_{v'}
        if row_layer == 1:
            # Pass (graph_idx, vertex) and (graph_idx, edge) tuples
            result = self._compute_q((row_graph, row_vertex), (col_graph, edge_w), row_graph)
            # Cache and return
            self.matrix_entries[cache_key] = result
            return result

        # Layer s > 1: Recursive case (equation 5.16, second line)
        # a^{s+1}_{v,w} = ⃗h₁(a^s_{v,w}) + Σ_{l∈E(Γ)} a^s_{v,l} q_{lw}
        graph = self.graphs[row_graph]

        # First, get a^s_{v,w} (previous layer)
        prev_layer_spec = (row_graph, row_vertex, row_layer - 1)
        col_spec_w = ('edge', row_graph, edge_w)
        a_prev = self.build_matrix_entry(prev_layer_spec, col_spec_w)

        # Compute ⃗h₁(a^s_{v,w})
        h_term = self._apply_h(a_prev, row_graph)
        # Extract principal part (vertex degree 1)
        h1_term = self._extract_principal_part(h_term, 1)

        # Principal part note:
        # a^s_{v,l} has vertex degree 1 (principal part), and q_{l,w} has degree 1,
        # so a^s_{v,l} * q_{l,w} has degree 2 and is removed by degree-1 extraction.
        # Therefore the Σ_l a^s_{v,l} q_{l,w} term does not contribute to the
        # principal part of A. We skip computing it for efficiency.
        result = h1_term

        # Simplify for deeper layers to prevent expression growth
        if self.use_symbolic and row_layer > 2:
            result = self._smart_simplify(result)

        # For the principal part of A, extract vertex degree 1
        if self.use_symbolic:
            result = self._extract_principal_part(result, 1)

        # Cache and return
        self.matrix_entries[cache_key] = result
        return result

    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Get statistics about cached computations.

        Returns:
            Dictionary with cache statistics including:
            - matrix_entries: Number of cached matrix entries
            - structure_functions: Number of cached structure functions
            - expanded_expressions: Number of cached expanded expressions
            - h_action_built: Whether Hamiltonian action has been built
        """
        stats = {
            'matrix_entries': len(self.matrix_entries),
            'structure_functions': len(self.structure_functions_symbolic),
            'expanded_expressions': len(self._expanded_expr_cache),
            'h_action_built': self._h_action_cache is not None
        }
        return stats

    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance timing statistics.

        Returns:
            Dictionary with timing statistics for each operation including:
            - total_time: Total time spent in this operation
            - call_count: Number of times this operation was called
            - avg_time: Average time per call
        """
        stats = {}
        for key in self._timing_stats.keys():
            total_time = self._timing_stats[key]
            count = self._timing_counts[key]
            avg_time = total_time / count if count > 0 else 0.0
            stats[key] = {
                'total_time': total_time,
                'call_count': count,
                'avg_time': avg_time
            }
        return stats

    def reset_timing_statistics(self):
        """Reset all performance timing statistics to zero."""
        for key in self._timing_stats.keys():
            self._timing_stats[key] = 0.0
            self._timing_counts[key] = 0

    def print_performance_report(self):
        """Print a formatted performance report with timing and cache statistics."""
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)

        # Cache statistics
        print("\nCache Statistics:")
        print("-" * 60)
        cache_stats = self.get_cache_statistics()
        print(f"  Matrix entries cached:      {cache_stats['matrix_entries']}")
        print(f"  Structure functions cached: {cache_stats['structure_functions']}")
        print(f"  Expanded expressions:       {cache_stats['expanded_expressions']}")
        print(f"  H-action built:             {cache_stats['h_action_built']}")

        # Timing statistics
        print("\nTiming Statistics:")
        print("-" * 60)
        timing_stats = self.get_timing_statistics()
        print(f"{'Operation':<30} {'Calls':<10} {'Total (s)':<12} {'Avg (s)':<12}")
        print("-" * 60)
        for op, stats in sorted(timing_stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
            if stats['call_count'] > 0:
                print(f"{op:<30} {stats['call_count']:<10} {stats['total_time']:<12.4f} {stats['avg_time']:<12.6f}")

        print("=" * 60 + "\n")

    def clear_cache(self, clear_structure_functions=False):
        """
        Clear all cached computations.

        This method resets the matrix entries cache and other cached values,
        forcing all subsequent calculations to be recomputed from scratch.

        Use this when:
        - Structure functions have been modified (set clear_structure_functions=True)
        - Alpha values have been changed
        - You want to ensure fresh calculations

        Args:
            clear_structure_functions: If True, also clear structure function cache

        Note: Under normal usage, you should NOT need to call this method.
        The cache is designed to correctly handle all calculations.
        """
        self.matrix_entries = {}
        # Clear other caches
        if hasattr(self, '_h_action_cache'):
            self._h_action_cache = None
        if hasattr(self, '_expanded_expr_cache'):
            self._expanded_expr_cache = {}
        # Optionally clear structure functions (usually not needed)
        if clear_structure_functions:
            self.structure_functions_symbolic = {}
        # Clear q cache
        if hasattr(self, '_q_cache'):
            self._q_cache = {}

def format_latex(expr, inline=True):
    """
    Format a sympy expression as LaTeX.

    Args:
        expr: Sympy expression or numeric value
        inline: If True, wrap in $...$ for inline LaTeX; if False, use $$...$$ for display mode

    Returns:
        LaTeX formatted string
    """
    if expr == 0:
        return "$0$" if inline else "$$0$$"

    if isinstance(expr, (int, float)):
        delim = "$" if inline else "$$"
        return f"{delim}{expr}{delim}"

    # Convert to LaTeX using sympy
    latex_str = latex(expr)

    # Replace "alpha" with Greek symbol "\alpha"
    # Need to be careful to only replace the symbol name, not "alpha" in other contexts
    import re
    # Match "alpha" followed by underscore (subscript) - this catches our alpha symbols
    latex_str = re.sub(r'\balpha_', r'\\alpha_', latex_str)
    # Also match standalone "alpha" (edge case)
    latex_str = re.sub(r'\balpha\b', r'\\alpha', latex_str)

    # Wrap in appropriate delimiters
    if inline:
        return f"${latex_str}$"
    else:
        return f"$${latex_str}$$"


def interactive_calculator():
    """Interactive interface for the FAS minor calculator."""

    print("=" * 60)
    print("Fundamental Algebraic System (FAS) Minor Calculator")
    print("=" * 60)
    print()

    # Input number of component graphs
    # Note: Single component case is trivial, need at least 2 components
    num_graphs = int(input("Enter number of component graphs (must be >= 2): "))
    if num_graphs < 2:
        print("WARNING: Single component case is trivial. Using at least 2 components is recommended.")
    graphs = []

    for i in range(num_graphs):
        print(f"\n--- Component Graph {i} ---")
        vertices_input = input(f"Enter vertices for graph {i} (comma-separated): ")
        vertices = [int(v.strip()) for v in vertices_input.split(',')]

        edges_input = input(f"Enter edges for graph {i} (format: 'u1,v1;u2,v2;...'): ")
        edges = []
        if edges_input.strip():
            for edge_str in edges_input.split(';'):
                u, v = edge_str.split(',')
                edges.append((int(u.strip()), int(v.strip())))

        graph = ComponentGraph(vertices, edges)
        graphs.append(graph)
        print(f"Created: {graph}")

    # Calculate n (dimension of manifold) and m
    total_vertices = sum(len(g.vertices) for g in graphs)
    total_edges = sum(len(g.edges) for g in graphs)
    n = total_vertices + total_edges  # Dimension of the manifold
    num_components = len(graphs)

    # For the FAS, we need to specify what m is based on the system
    # Typically m relates to the number of constraints or independent equations
    print(f"\nTotal vertices: {total_vertices}")
    print(f"Total edges: {total_edges}")
    print(f"Manifold dimension (n = vertices + edges): {n}")
    print(f"Number of components: {num_components}")

    # User should specify m based on their system
    m = int(input("Enter the value of m (number of constraints): "))
    num_rows = n - m + 1
    print(f"Required rows (n-m+1): {num_rows}")

    # Always use symbolic computation (numeric not supported)
    print("\nNote: Only symbolic computation is supported. Using symbolic mode.")
    calc = FASMinorCalculator(graphs, use_symbolic=True)

    # Input row specifications
    print(f"\nEnter {num_rows} row specifications in (graph_idx, vertex, layer) format:")
    row_specs = []
    for i in range(num_rows):
        row_input = input(f"Row {i+1} (format: 'graph_idx,vertex,layer'): ")
        g_idx, vertex, layer = [int(x.strip()) for x in row_input.split(',')]
        row_specs.append((g_idx, vertex, layer))

    print(f"\nRow specifications: {row_specs}")

    # Symbolic-only; no numeric structure-function input is supported

    # Calculate determinant (minor)
    print("\nCalculating minor (determinant)...")
    try:
        from determinant_computer import DeterminantComputer
    except Exception as e:
        print(f"Error: could not import DeterminantComputer: {e}")
        return

    det_comp = DeterminantComputer(calc)

    try:
        det_val = det_comp.compute_determinant(row_specs)
    except Exception as e:
        print(f"Error computing determinant: {e}")
        return

    print(f"\n{'=' * 60}")
    print(f"RESULT: Minor = {format_latex(det_val, inline=False)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Check if sympy is available
    try:
        import sympy
    except ImportError:
        print("Warning: sympy not found. Symbolic mode will not be available.")
        print("Install with: pip install sympy")
        print()

    interactive_calculator()

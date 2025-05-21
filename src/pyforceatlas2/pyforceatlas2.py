"""
pyforceatlas2/pyforceatlas2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main module implementing the ForceAtlas2 graph layout algorithm.

References:
    Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).
    ForceAtlas2, a continuous graph layout algorithm for handy network visualization
    designed for the Gephi software. PLoS ONE, 9(6), e98679.
    https://doi.org/10.1371/journal.pone.0098679
"""

import random

import igraph
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyforceatlas2 import layout


class ForceAtlas2:
    """
    Implementation of the ForceAtlas2 layout algorithm, wrapping the functions in `layout.py`.

    This class orchestrates the overall layout process:
      - Builds node/edge structures from an adjacency matrix (or from NetworkX/igraph).
      - Runs iterative repulsion, gravity, and attraction steps.
      - Adjusts speed each iteration for a continuous layout experience.

    Args:
        outbound_attraction_distribution (bool):
            If True, distributes attraction along outbound edges to dissuade hubs.
            In practice, this divides the attraction force by node mass, so
            nodes with high degree do not pull as strongly on their neighbors.
        edge_weight_influence (float):
            Exponent controlling how edge weights affect attraction.
            - 0 => ignore weights (treat all edges as weight 1).
            - 1 => use weights as-is.
            - >1 => amplify differences in weights.
            Typical range: [0, 2].
        lin_log_mode (bool):
            If True, use LinLog attraction, i.e. F ~ log(1 + dist)/dist,
            which tends to tighten clusters and highlight community structure.
            Otherwise, use linear attraction.
        jitter_tolerance (float):
            Tolerance parameter controlling how quickly the layout moves.
            Lower => more stable but slower movement; higher => faster, riskier.
            Typical range: [0.1, 10].
        barnes_hut_optimize (bool):
            If True, use Barnes-Hut optimization for repulsion, giving O(n log n) complexity.
            Recommended for graphs > 1,000 nodes.
        barnes_hut_theta (float):
            Barnes-Hut parameter controlling approximation accuracy vs. speed.
            Typical range: [0.5, 2.0].
            Lower => more accurate, slower. Higher => faster, less accurate.
        scaling_ratio (float):
            Scales the overall repulsion strength. Higher => more spread out.
            Typical range: [1.0, 10.0].
        strong_gravity_mode (bool):
            If True, applies a stronger gravity force (linear w.r.t. distance)
            that can keep large components from drifting away.
        gravity (float):
            The gravitational constant. 0.0 => no gravity, ~1.0 => moderate pull.
            Typical range: [0.0, 5.0].
        verbose (bool):
            If True, displays iteration progress via tqdm.
    """

    def __init__(
        self,
        outbound_attraction_distribution=False,
        edge_weight_influence=1.0,
        lin_log_mode=False,
        jitter_tolerance=1.0,
        barnes_hut_optimize=True,
        barnes_hut_theta=1.2,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        verbose=True,
    ):
        self.outbound_attraction_distribution = outbound_attraction_distribution
        self.edge_weight_influence = edge_weight_influence
        self.lin_log_mode = lin_log_mode
        self.jitter_tolerance = jitter_tolerance
        self.barnes_hut_optimize = barnes_hut_optimize
        self.barnes_hut_theta = barnes_hut_theta
        self.scaling_ratio = scaling_ratio
        self.strong_gravity_mode = strong_gravity_mode
        self.gravity = gravity
        self.verbose = verbose

    def init(self, graph, pos=None):
        """
        Initialize nodes and edges from the input graph (NumPy array or SciPy sparse matrix).

        - Checks that `graph` is square and symmetric (undirected).
        - Builds a list of `Node` objects, assigning mass = (degree + 1).
        - Builds a list of `Edge` objects from nonzero entries in the adjacency matrix.
        - If `pos` is not provided, random (x,y) coordinates are assigned.

        Args:
            graph (np.ndarray or scipy.sparse.spmatrix):
                The adjacency matrix for the undirected graph.
                Must be square, typically symmetric.
            pos (np.ndarray, optional):
                Initial positions (shape = [num_nodes, 2]).
                If None, positions are randomized.

        Returns:
            tuple:
                (nodes, edges)
                - nodes: list of layout.Node objects
                - edges: list of layout.Edge objects

        Raises:
            ValueError: If graph is not a valid NumPy or SciPy structure, or not square.
        """
        # Check graph type and shape
        is_sparse = False
        if isinstance(graph, np.ndarray):
            assert graph.shape[0] == graph.shape[1], "Graph is not a square matrix."
            assert np.all(
                graph.T == graph
            ), "Graph matrix is not symmetric (undirected graphs only)."
            assert isinstance(pos, np.ndarray) or pos is None, "Invalid node positions."
        elif scipy.sparse.issparse(graph):
            assert graph.shape[0] == graph.shape[1], "Graph is not a square matrix."
            assert isinstance(pos, np.ndarray) or pos is None, "Invalid node positions."
            graph = graph.tolil()  # LIL format for efficient row access
            is_sparse = True
        else:
            raise ValueError("Graph must be a NumPy array or a SciPy sparse matrix.")

        # Initialize nodes with mass = (degree + 1)
        nodes = []
        num_nodes = graph.shape[0]
        for i in range(num_nodes):
            node = layout.Node()
            # Set mass = (degree + 1)
            if is_sparse:
                node.mass = 1 + len(graph.rows[i])
            else:
                node.mass = 1 + np.count_nonzero(graph[i])

            # Initialize forces to zero
            node.old_dx = 0.0
            node.old_dy = 0.0
            node.dx = 0.0
            node.dy = 0.0
            # Assign positions
            if pos is None:
                node.x = random.random()
                node.y = random.random()
            else:
                node.x = pos[i][0]
                node.y = pos[i][1]

            nodes.append(node)

        # Build edges from nonzero adjacency entries
        edges = []
        edge_indices = np.asarray(graph.nonzero()).T
        for edge_index in edge_indices:
            if edge_index[1] <= edge_index[0]:
                # Skip duplicates (only handle upper triangle for undirected)
                continue

            edge = layout.Edge()
            edge.node1 = edge_index[0]
            edge.node2 = edge_index[1]
            edge.weight = graph[tuple(edge_index)]
            edges.append(edge)

        # Cache flat edge arrays for vectorized kernels
        rows, cols = graph.nonzero()
        mask = cols > rows
        self._edge_i = rows[mask]
        self._edge_j = cols[mask]
        # use .A1 for sparse, else direct indexing
        vals = graph[self._edge_i, self._edge_j]
        self._edge_w = (vals.A1 if hasattr(vals, "A1") else vals).astype(float)

        return nodes, edges

    def forceatlas2(self, graph, pos=None, iterations=100):
        """
        Compute node positions using the ForceAtlas2 layout algorithm.

        Iteratively:
          1. Reset forces to 0.
          2. (Optional) Build a Barnes-Hut tree if enabled and apply approximate repulsion.
             Else, apply O(n^2) repulsion.
          3. Apply gravity (linear or strong).
          4. Apply attraction along edges (lin-log or linear).
          5. Adjust speed based on jitter, apply forces to node positions.

        Args:
            graph (np.ndarray or scipy.sparse.spmatrix):
                The adjacency matrix for the undirected graph.
            pos (np.ndarray, optional):
                Initial positions, shape [num_nodes, 2]. If None, random init.
            iterations (int, optional):
                Number of layout iterations (a typical range is 50–1000,
                depending on graph size and desired precision).

        Returns:
            list of tuple:
                A list of (x, y) positions for each node, in node order.

        Raises:
            ValueError: If the graph is invalid or not square.
        """
        # Initialize parameters and the node/edge structures
        speed = 1.0
        speed_efficiency = 1.0
        nodes, edges = self.init(graph, pos)

        # If distributing attraction, we compute a reference "outbound_att_comp"
        # typically the average node mass.
        outbound_att_comp = 1.0
        if self.outbound_attraction_distribution:
            outbound_att_comp = np.mean([node.mass for node in nodes])

        # Set up iteration range
        iter_range = range(iterations)
        if self.verbose:
            iter_range = tqdm(iter_range, desc="ForceAtlas2 iterations", mininterval=0.1)

        for _ in iter_range:
            # Store old forces, reset current
            for node in nodes:
                node.old_dx = node.dx
                node.old_dy = node.dy
                node.dx = 0.0
                node.dy = 0.0

            # Build Barnes-Hut tree if enabled, then apply repulsion
            if self.barnes_hut_optimize:
                root_region = layout.Region(nodes)
                root_region.build_subregions()
                root_region.apply_force_on_nodes(nodes, self.barnes_hut_theta, self.scaling_ratio)
            else:
                # For small graphs, we can do pairwise repulsion
                layout.apply_repulsion(nodes, self.barnes_hut_theta, self.scaling_ratio)

            # Apply gravity (strong or linear)
            layout.apply_gravity(
                nodes,
                gravity=self.gravity,
                scaling_ratio=self.scaling_ratio,
                use_strong_gravity=self.strong_gravity_mode,
            )
            # Apply attraction along edges
            layout.apply_attraction(
                nodes,
                edges,
                distributed_attraction=self.outbound_attraction_distribution,
                coefficient=outbound_att_comp,  # If OAD is enabled, we scale by average mass
                edge_weight_influence=self.edge_weight_influence,
                lin_log_mode=self.lin_log_mode,
            )

            # Adjust speed and move nodes
            speed_vals = layout.adjust_speed_and_apply_forces(
                nodes, speed, speed_efficiency, self.jitter_tolerance
            )
            speed = speed_vals["speed"]
            speed_efficiency = speed_vals["speed_efficiency"]

        return [(node.x, node.y) for node in nodes]

    def forceatlas2_networkx_layout(self, graph, pos=None, iterations=100, weight_attr=None):
        """
        Compute a layout for a NetworkX graph using the ForceAtlas2 algorithm.

        Converts the NetworkX graph to a SciPy sparse matrix, then calls `forceatlas2()`.

        Args:
            graph (networkx.Graph):
                A NetworkX undirected graph.
            pos (dict, optional):
                Initial positions as {node: (x, y)}. If None, random init is used.
            iterations (int, optional):
                Number of layout iterations. Default 100.
            weight_attr (str, optional):
                Name of the edge attribute to use as weight. If None, all edges = 1.

        Returns:
            dict:
                A mapping {node: (x, y)} for all nodes in the graph.
        """
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph (undirected).")
        assert (
            isinstance(pos, dict) or pos is None
        ), "Positions must be a dict {node: (x,y)} or None."

        # Convert to SciPy sparse adjacency
        sparse_matrix = nx.to_scipy_sparse_array(graph, dtype="f", format="lil", weight=weight_attr)
        if pos is None:
            layout_coords = self.forceatlas2(sparse_matrix, pos=None, iterations=iterations)
        else:
            # Convert pos dict to array, preserving the graph's node ordering
            pos_array = np.asarray([pos[node] for node in graph.nodes()])
            layout_coords = self.forceatlas2(sparse_matrix, pos=pos_array, iterations=iterations)

        # Build dict {node: (x, y)}
        return dict(zip(graph.nodes(), layout_coords))

    def forceatlas2_igraph_layout(self, graph, pos=None, iterations=100, weight_attr=None):
        """
        Compute a layout for an igraph.Graph using the ForceAtlas2 algorithm.

        1. Convert igraph to a SciPy sparse adjacency.
        2. Optionally supply initial positions.
        3. Call `forceatlas2()`.
        4. Return an igraph.Layout object.

        Args:
            graph (igraph.Graph):
                An undirected igraph graph.
            pos (list or np.ndarray, optional):
                Initial positions. shape = [num_nodes, 2].
                If None, random init is used.
            iterations (int, optional):
                Number of layout iterations. Default 100.
            weight_attr (str, optional):
                Name of the edge attribute for weight. If None, all edges = 1.

        Returns:
            igraph.Layout:
                The final layout as an igraph.Layout object.
        """

        def to_sparse(ig_graph, weight_attr=None):
            """
            Convert an igraph Graph to a SciPy CSR sparse matrix.

            - If undirected, each edge is duplicated (u->v and v->u).
            - If weight_attr is None, weights default to 1.
            """
            edges = ig_graph.get_edgelist()
            if weight_attr is None:
                weights = [1] * len(edges)
            else:
                weights = ig_graph.es[weight_attr]

            # Duplicate edges if undirected
            if not ig_graph.is_directed():
                edges.extend([(v, u) for (u, v) in edges])
                weights.extend(weights)

            return csr_matrix((weights, list(zip(*edges))))

        if not isinstance(graph, igraph.Graph):
            raise ValueError("Input must be an igraph.Graph (undirected).")
        assert (
            isinstance(pos, (list, np.ndarray)) or pos is None
        ), "Positions must be a list/array of shape [num_nodes, 2] or None."

        if isinstance(pos, list):
            pos = np.array(pos)

        # Build adjacency matrix
        adj_matrix = to_sparse(graph, weight_attr)
        # Compute layout coords
        coordinates = self.forceatlas2(adj_matrix, pos=pos, iterations=iterations)

        # Return as igraph.Layout
        return igraph.Layout(coordinates, dim=2)

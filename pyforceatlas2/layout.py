"""
pyforceatlas2/layout
~~~~~~~~~~~~~~~~~~~~

An implementation of the ForceAtlas2 graph layout algorithm.

References:
    - Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014). 
      ForceAtlas2, a continuous graph layout algorithm for handy network visualization 
      designed for the Gephi software. PLoS ONE, 9(6), e98679.
      https://doi.org/10.1371/journal.pone.0098679
"""

from math import log, sqrt


class Node:
    """Represents a node in the ForceAtlas2 algorithm.

    Attributes:
        mass (float):
            Node mass, typically set to (degree + 1) if you want
            "degree-based repulsion." This ensures isolated nodes
            still have nonzero mass.
        old_dx (float):
            Previous step's x-axis force (used to compute "swing" and traction).
        old_dy (float):
            Previous step's y-axis force (used to compute "swing" and traction).
        dx (float):
            Accumulated x-axis force in the current iteration.
        dy (float):
            Accumulated y-axis force in the current iteration.
        x (float):
            Node's x-coordinate in the 2D layout space.
        y (float):
            Node's y-coordinate in the 2D layout space.
    """

    def __init__(self):
        self.mass = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0


class Edge:
    """Represents an edge connecting two nodes in the ForceAtlas2 algorithm.

    Attributes:
        node1 (int):
            Index of the first node in the nodes list.
        node2 (int):
            Index of the second node in the nodes list.
        weight (float):
            Weight of the edge, used for weighted attraction.
    """

    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


def lin_repulsion(node_a, node_b, coefficient=0.0):
    """Apply a linear (Fruchterman-Reingold style) repulsion force between two nodes.

    F_rep ~ coefficient * (mass_a * mass_b) / distance^2

    Args:
        node_a (Node): First node.
        node_b (Node): Second node.
        coefficient (float, optional):
            Repulsion coefficient. Higher values push nodes apart more strongly.
            Typical range: 1.0 to 10.0 (or higher for very dense networks).
    """
    # Compute distance between nodes
    dx = node_a.x - node_b.x
    dy = node_a.y - node_b.y
    dist_sq = dx * dx + dy * dy
    if dist_sq > 0:
        factor = coefficient * node_a.mass * node_b.mass / dist_sq
        node_a.dx += dx * factor
        node_a.dy += dy * factor
        node_b.dx -= dx * factor
        node_b.dy -= dy * factor


def lin_repulsion_region(node, region, coefficient=0.0):
    """Apply an approximate repulsion force from a region's center of mass (Barnes-Hut).

    Instead of pairwise repulsions for all nodes, we approximate
    a group of distant nodes by their center of mass.

    Args:
        node (Node): Node receiving force.
        region (Region): Region object with mass, mass_center_x, mass_center_y.
        coefficient (float, optional):
            Repulsion coefficient (same meaning as in lin_repulsion).
    """
    # Compute distance to region's center of mass
    dx = node.x - region.mass_center_x
    dy = node.y - region.mass_center_y
    dist_sq = dx * dx + dy * dy
    if dist_sq > 0:
        factor = coefficient * node.mass * region.mass / dist_sq
        node.dx += dx * factor
        node.dy += dy * factor


def lin_gravity(node, gravity):
    """Apply a standard gravitational force that attracts the node toward the origin (0, 0).

    F_gravity ~ node.mass * gravity

    This is a simple linear pull toward (0,0) and helps keep the layout
    from drifting. If your data has disconnected components, gravity
    prevents them from flying away.

    Args:
        node (Node): The node to which gravity is applied.
        gravity (float):
            Gravitational constant. Typical range: 0.0 to 5.0.
            If 0.0, there's effectively no gravity.
    """
    # Compute distance to origin
    dx = node.x
    dy = node.y
    distance = sqrt(dx * dx + dy * dy)
    if distance > 0:
        factor = node.mass * gravity / distance
        node.dx -= dx * factor
        node.dy -= dy * factor


def strong_gravity(node, gravity, coefficient=0.0):
    """Apply a 'strong gravity' force to a node, pulling it more aggressively to (0,0).

    F_strong_gravity ~ coefficient * node.mass * gravity * distance

    Unlike linear gravity, strong gravity grows linearly with distance,
    causing large clusters to be pulled in more strongly. Useful if
    your layout is "too spread out" and you need a compact shape.

    Args:
        node (Node): Node to which strong gravity is applied.
        gravity (float):
            Gravitational constant, typically 1.0 to 5.0.
        coefficient (float, optional):
            Additional scaling factor, e.g. 1.0 to 5.0.
            Increase to force a more compact layout.
    """
    # Skip the origin (0,0) to avoid division by zero
    if node.x != 0 or node.y != 0:
        factor = coefficient * node.mass * gravity
        node.dx -= node.x * factor
        node.dy -= node.y * factor


def lin_attraction(
    node_a, node_b, edge_weight, distributed_attraction, coefficient=0.0, lin_log_mode=False
):
    """Apply an attractive force between two connected nodes.

    - In linear mode:
        F_attr ~ -coefficient * edge_weight
    - In LinLog mode:
        F_attr ~ -coefficient * edge_weight * (log(1 + distance) / distance)

    If distributed_attraction=True, the force is divided by node_a's mass,
    which is often used to 'dissuade hubs' or reduce the pull of high-degree nodes.

    Args:
        node_a (Node): The first node.
        node_b (Node): The second node.
        edge_weight (float):
            The weight of the edge connecting the two nodes.
            If edges are unweighted, this is typically 1.0.
        distributed_attraction (bool):
            If True, divides the attraction by node_a.mass.
            Helps limit the pull from nodes with large mass.
        coefficient (float, optional):
            Overall attraction coefficient. Typical range: 1.0 to 10.0.
        lin_log_mode (bool, optional):
            If True, uses log(1 + distance)/distance for the attraction,
            highlighting clusters more strongly (LinLog model).
    """
    # Compute the distance between nodes
    dx = node_a.x - node_b.x
    dy = node_a.y - node_b.y

    if lin_log_mode:
        # LinLog mode: log(1 + distance) / distance
        distance = sqrt(dx * dx + dy * dy)
        if distance > 0:
            log_factor = log(1 + distance) / distance
            if not distributed_attraction:
                factor = -coefficient * edge_weight * log_factor
            else:
                factor = -coefficient * edge_weight * log_factor / node_a.mass
        else:
            # Nodes are at the same position; no attraction needed.
            factor = 0.0
    else:
        # Linear mode: constant factor
        if not distributed_attraction:
            factor = -coefficient * edge_weight
        else:
            factor = -coefficient * edge_weight / node_a.mass

    # Apply the force to both nodes
    node_a.dx += dx * factor
    node_a.dy += dy * factor
    node_b.dx -= dx * factor
    node_b.dy -= dy * factor


def apply_repulsion(nodes, coefficient):
    """Apply pairwise repulsion forces between all nodes (O(n^2)).

    For large graphs, you typically want Barnes-Hut approximation
    instead (using Region). But for smaller graphs (<= 1,000 nodes),
    this can be acceptable.

    Args:
        nodes (list of Node): All nodes in the graph.
        coefficient (float):
            Repulsion coefficient. Typically 1.0 to 10.0 or more
            for very dense networks.
    """
    # Apply repulsion forces between all nodes
    for i, node in enumerate(nodes):
        for other in nodes[:i]:
            lin_repulsion(node, other, coefficient)


def apply_gravity(nodes, gravity, scaling_ratio, use_strong_gravity=False):
    """Apply gravitational forces to all nodes.

    This either uses standard gravity or strong gravity
    depending on `use_strong_gravity`.

    Args:
        nodes (list of Node): All nodes in the graph.
        gravity (float):
            Gravitational constant.
            0.0 means no gravity,
            typical range is 0.1 to 5.0 or so.
        scaling_ratio (float):
            Used only if `use_strong_gravity` is True,
            effectively scales the strong gravity pull.
            (e.g. 1.0 to 5.0).
        use_strong_gravity (bool, optional):
            Whether to use strong gravity. Defaults to False.
            If True, pulls distant nodes in more aggressively.
    """
    # Apply gravity to all nodes
    for node in nodes:
        if use_strong_gravity:
            # scaling_ratio is used here as the coefficient
            strong_gravity(node, gravity, scaling_ratio)
        else:
            lin_gravity(node, gravity)


def apply_attraction(
    nodes, edges, distributed_attraction, coefficient, edge_weight_influence, lin_log_mode=False
):
    """Apply attractive forces along edges between nodes.

    The actual attraction formula is in `lin_attraction`.
    Here we handle the edge-weight influence exponent.

    Args:
        nodes (list of Node):
            All nodes in the graph.
        edges (list of Edge):
            Edges with references to node indices and weights.
        distributed_attraction (bool):
            If True, divide attraction by node mass (see lin_attraction).
        coefficient (float):
            Attraction coefficient, typically 1.0 to 10.0.
        edge_weight_influence (float):
            Exponent for edge weight.
            - 0 => all edges considered weight 1
            - 1 => direct use of edge.weight
            - >1 => amplify differences in edge weight
        lin_log_mode (bool, optional):
            If True, use LinLog mode for the attraction.
    """
    # Apply attraction forces along edges
    for edge in edges:
        if edge_weight_influence == 0:
            effective_weight = 1
        elif edge_weight_influence == 1:
            effective_weight = edge.weight
        else:
            effective_weight = pow(edge.weight, edge_weight_influence)

        # Apply attraction between nodes
        lin_attraction(
            nodes[edge.node1],
            nodes[edge.node2],
            effective_weight,
            distributed_attraction,
            coefficient,
            lin_log_mode,
        )


class Region:
    """Represents a spatial region for Barnes-Hut approximation of repulsion.

    Attributes:
        mass (float):
            Total mass of nodes in this region.
        mass_center_x (float):
            x-coordinate of the center of mass.
        mass_center_y (float):
            y-coordinate of the center of mass.
        size (float):
            An approximate bounding size of the region (used in BH test).
        nodes (list of Node):
            The actual nodes contained in this region.
        subregions (list of Region):
            Subdivided subregions if we recursively partition the space.
    """

    def __init__(self, nodes):
        self.mass = 0.0
        self.mass_center_x = 0.0
        self.mass_center_y = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.update_mass_and_geometry()

    def update_mass_and_geometry(self):
        """Update the region's mass, center of mass, and size
        based on the nodes it contains.

        Typically called after creating or subdividing the region.
        """
        # Update mass and center of mass
        if len(self.nodes) > 1:
            total_mass = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for node in self.nodes:
                total_mass += node.mass
                sum_x += node.x * node.mass
                sum_y += node.y * node.mass

            self.mass = total_mass
            self.mass_center_x = sum_x / total_mass
            self.mass_center_y = sum_y / total_mass

            # Compute the "size" of the region
            max_distance = 0.0
            for node in self.nodes:
                distance = sqrt(
                    (node.x - self.mass_center_x) ** 2 + (node.y - self.mass_center_y) ** 2
                )
                # '2 * distance' ensures we cover diameter, not just radius
                max_distance = max(max_distance, 2 * distance)

            self.size = max_distance

    def build_subregions(self):
        """Subdivide the region into up to four quadrants based on node positions.

        This is part of the standard Barnes-Hut oct/quadtree approach:
        - If there's more than one node, we recursively partition.
        - Each quadrant becomes a subregion with its own center of mass.
        """
        if len(self.nodes) <= 1:
            return

        # Split nodes into four quadrants based on mass center
        top_left, bottom_left = [], []
        top_right, bottom_right = [], []
        for node in self.nodes:
            if node.x < self.mass_center_x:
                if node.y < self.mass_center_y:
                    bottom_left.append(node)
                else:
                    top_left.append(node)
            else:
                if node.y < self.mass_center_y:
                    bottom_right.append(node)
                else:
                    top_right.append(node)

        # Create subregions for each quadrant
        quadrants = [top_left, bottom_left, top_right, bottom_right]
        for quadrant in quadrants:
            if quadrant:
                subregion = Region(quadrant)
                self.subregions.append(subregion)

        # Recursively build sub-subregions
        for subregion in self.subregions:
            subregion.build_subregions()

    def apply_force(self, node, theta, coefficient=0.0):
        """Apply repulsion force from this region onto a node using Barnes-Hut approximation.

        If (distance * theta) > self.size, we treat this region as a single
        entity (use lin_repulsion_region). Otherwise, we recurse into subregions.

        Args:
            node (Node): Node receiving the force.
            theta (float):
                Barnes-Hut parameter controlling accuracy vs speed.
                Typical range: 0.5 to 2.0.
                Lower => more accurate, slower.
                Higher => less accurate, faster.
            coefficient (float, optional):
                Repulsion coefficient, same usage as in lin_repulsion.
        """
        if len(self.nodes) < 2:
            # Only one node => direct pairwise repulsion
            lin_repulsion(node, self.nodes[0], coefficient)
        else:
            dx = node.x - self.mass_center_x
            dy = node.y - self.mass_center_y
            distance = sqrt(dx * dx + dy * dy)
            # If region is sufficiently far, approximate with center of mass
            if distance * theta > self.size:
                lin_repulsion_region(node, self, coefficient)
            else:
                # Otherwise, recurse deeper
                for subregion in self.subregions:
                    subregion.apply_force(node, theta, coefficient)

    def apply_force_on_nodes(self, nodes, theta, coefficient=0.0):
        """Apply Barnes-Hut repulsion forces on a list of nodes,
        iterating over each node and calling apply_force.

        Args:
            nodes (list of Node):
                Nodes to receive force.
            theta (float):
                Barnes-Hut approximation parameter.
            coefficient (float, optional):
                Repulsion coefficient.
        """
        for node in nodes:
            self.apply_force(node, theta, coefficient)


def adjust_speed_and_apply_forces(nodes, speed, speed_efficiency, jitter_tolerance):
    """Adjust simulation speed and update node positions based on the accumulated forces.

    This step is crucial for a 'continuous' layout:
    - We compute each node's "swing" (difference in direction from previous iteration).
    - We compute "traction" (overall magnitude of forces).
    - Then adapt the global speed to keep swinging under control.
    - Finally, apply the forces to update (x,y) positions.

    Args:
        nodes (list of Node):
            Nodes in the simulation.
        speed (float):
            Current global speed of the simulation (step size).
        speed_efficiency (float):
            Efficiency factor for speed adjustment, can go up or down
            depending on how stable the layout is.
            Typical range: starts around 1.0, can drift between [0.05, 10].
        jitter_tolerance (float):
            Tolerance for "oscillations." Lower => more stable but slower movement.
            Typical range: 0.1 to 10. Higher values => faster, less stable.

    Returns:
        dict:
            A dictionary with updated 'speed' and 'speed_efficiency'.
            Example: {"speed": new_speed, "speed_efficiency": new_efficiency}
    """
    # Compute global swinging and traction
    total_swing = 0.0
    total_traction = 0.0
    for node in nodes:
        delta_dx = node.old_dx - node.dx
        delta_dy = node.old_dy - node.dy
        swing = sqrt(delta_dx * delta_dx + delta_dy * delta_dy)
        total_swing += node.mass * swing
        # Traction is proportional to the "speed" of the node
        traction = (
            0.5 * node.mass * sqrt((node.old_dx + node.dx) ** 2 + (node.old_dy + node.dy) ** 2)
        )
        total_traction += traction

    # Basic jitter estimation
    estimated_jitter = 0.05 * sqrt(len(nodes))
    min_jitter = sqrt(estimated_jitter)
    max_jitter = 10
    # jt is the "actual" jitter threshold
    jt = jitter_tolerance * max(
        min_jitter, min(max_jitter, estimated_jitter * total_traction / (len(nodes) ** 2))
    )

    min_speed_efficiency = 0.05
    if total_traction and (total_swing / total_traction > 2.0):
        # If there's too much swinging, reduce efficiency
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.5
        jt = max(jt, jitter_tolerance)

    # target_speed: how fast we want to go, based on ratio of traction to swing
    if total_swing == 0:
        target_speed = float("inf")  # No movement => can accelerate freely
    else:
        target_speed = jt * speed_efficiency * total_traction / total_swing

    # More adjustments if we overshoot
    if total_swing > jt * total_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.7
    elif speed < 1000:
        speed_efficiency *= 1.3

    # Limit how quickly speed can rise
    max_rise = 0.5
    speed += min(target_speed - speed, max_rise * speed)

    # Finally, update positions
    for node in nodes:
        # "Local" damping factor
        delta_dx = node.old_dx - node.dx
        delta_dy = node.old_dy - node.dy
        swing = node.mass * sqrt(delta_dx * delta_dx + delta_dy * delta_dy)
        factor = speed / (1.0 + sqrt(speed * swing))

        node.x += node.dx * factor
        node.y += node.dy * factor

    return {"speed": speed, "speed_efficiency": speed_efficiency}

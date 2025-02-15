"""
tests/test_layout.py
~~~~~~~~~~~~~~~~~~~~~
"""

import math

import pytest

from pyforceatlas2 import layout

###############################################################################
# 1. Tests for basic classes and force functions in layout.py
###############################################################################


def test_node_initialization():
    """Test that a Node object initializes with default values."""
    node = layout.Node()
    assert node.mass == 0.0
    assert node.old_dx == 0.0
    assert node.old_dy == 0.0
    assert node.dx == 0.0
    assert node.dy == 0.0
    assert node.x == 0.0
    assert node.y == 0.0


def test_edge_initialization():
    """Test that an Edge object initializes with default values."""
    edge = layout.Edge()
    assert edge.node1 == -1
    assert edge.node2 == -1
    assert edge.weight == 0.0


def test_lin_repulsion():
    """Test linear repulsion between two nodes."""
    # Two nodes at (1,1) and (2,2) with mass 2 each.
    a = layout.Node()
    b = layout.Node()
    a.mass = 2.0
    b.mass = 2.0
    a.x, a.y = 1.0, 1.0
    b.x, b.y = 2.0, 2.0

    # Reset forces
    a.dx = a.dy = b.dx = b.dy = 0.0

    layout.lin_repulsion(a, b, coefficient=1.0)
    # dx = 1-2 = -1, dy = -1, dist_sq = 2, factor = (1*2*2)/2 = 2.
    # Expected update: a.dx += -1*2 = -2, a.dy += -1*2 = -2,
    #                   b.dx -= -1*2 = 2,  b.dy -= -1*2 = 2.
    assert a.dx == pytest.approx(-2.0)
    assert a.dy == pytest.approx(-2.0)
    assert b.dx == pytest.approx(2.0)
    assert b.dy == pytest.approx(2.0)


def test_lin_repulsion_no_update_for_zero_distance():
    """Test that no update occurs when two nodes are at the same position."""
    # When two nodes are at the same position, no update should occur.
    a = layout.Node()
    b = layout.Node()
    a.mass = 2.0
    b.mass = 3.0
    a.x = b.x = 1.0
    a.y = b.y = 1.0
    a.dx = a.dy = b.dx = b.dy = 0.0
    layout.lin_repulsion(a, b, coefficient=1.0)
    assert a.dx == 0.0
    assert a.dy == 0.0
    assert b.dx == 0.0
    assert b.dy == 0.0


def test_lin_repulsion_region():
    """Test linear repulsion between a node and a region."""
    # Test approximate repulsion from a region.
    node = layout.Node()
    node.mass = 3.0
    node.x = 5.0
    node.y = 5.0
    node.dx = node.dy = 0.0

    # Create a dummy region using a single node.
    dummy = layout.Node()
    dummy.mass = 2.0
    dummy.x = 3.0
    dummy.y = 3.0
    region = layout.Region([dummy])
    # For our test, force the region's mass and center manually.
    region.mass = 2.0
    region.mass_center_x = 3.0
    region.mass_center_y = 3.0

    layout.lin_repulsion_region(node, region, coefficient=1.0)
    # dx = 5-3 = 2, dy = 2, dist_sq = 8, factor = 1*3*2/8 = 0.75.
    # Expected: node.dx += 2*0.75 = 1.5, node.dy += 1.5.
    assert node.dx == pytest.approx(1.5)
    assert node.dy == pytest.approx(1.5)


def test_lin_gravity():
    """Test linear gravity."""
    node = layout.Node()
    node.mass = 2.0
    node.x = 3.0
    node.y = 4.0  # distance = 5
    node.dx = node.dy = 0.0

    layout.lin_gravity(node, gravity=1.0)
    # factor = (2*1)/5 = 0.4, so: dx -= 3*0.4 = 1.2, dy -= 4*0.4 = 1.6.
    assert node.dx == pytest.approx(-1.2)
    assert node.dy == pytest.approx(-1.6)


def test_strong_gravity():
    """Test strong gravity."""
    node = layout.Node()
    node.mass = 2.0
    node.x = 3.0
    node.y = 4.0
    node.dx = node.dy = 0.0

    layout.strong_gravity(node, gravity=1.0, coefficient=1.0)
    # factor = 1*2*1 = 2, so: dx -= 3*2 = 6, dy -= 4*2 = 8.
    assert node.dx == pytest.approx(-6.0)
    assert node.dy == pytest.approx(-8.0)


def test_lin_attraction_linear():
    """Test linear attraction between two nodes."""
    a = layout.Node()
    b = layout.Node()
    a.mass = 2.0
    b.mass = 3.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 3.0, 4.0  # distance = 5
    a.dx = a.dy = b.dx = b.dy = 0.0
    edge_weight = 1.0
    # Using linear mode with distributed_attraction=False.
    layout.lin_attraction(
        a, b, edge_weight, distributed_attraction=False, coefficient=2.0, lin_log_mode=False
    )
    # dx = 0-3 = -3; factor = -2.0 (since coefficient=2.0 and no division by mass).
    # a.dx += (-3)*(-2) = 6, a.dy += (-4)*(-2) = 8.
    # b.dx gets the opposite: -6 and -8.
    assert a.dx == pytest.approx(6.0)
    assert a.dy == pytest.approx(8.0)
    assert b.dx == pytest.approx(-6.0)
    assert b.dy == pytest.approx(-8.0)


def test_lin_attraction_linlog():
    """Test linear-logarithmic attraction between two nodes."""
    a = layout.Node()
    b = layout.Node()
    a.mass = 2.0
    b.mass = 3.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 3.0, 4.0
    a.dx = a.dy = b.dx = b.dy = 0.0
    edge_weight = 1.0
    # In lin-log mode, factor = -coefficient * edge_weight * (log(1+distance)/distance)
    # For distance=5: log(6)/5.
    expected_factor = -2.0 * (math.log(6) / 5)
    layout.lin_attraction(
        a, b, edge_weight, distributed_attraction=False, coefficient=2.0, lin_log_mode=True
    )
    # dx = 0-3 = -3 and dy = -4.
    assert a.dx == pytest.approx(-3 * expected_factor)
    assert a.dy == pytest.approx(-4 * expected_factor)
    assert b.dx == pytest.approx(3 * expected_factor)
    assert b.dy == pytest.approx(4 * expected_factor)


def test_apply_repulsion():
    """Test repulsion between two nodes."""
    # Two nodes placed 1 unit apart.
    a = layout.Node()
    b = layout.Node()
    a.mass = b.mass = 2.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 1.0, 0.0
    a.dx = a.dy = b.dx = b.dy = 0.0

    layout.apply_repulsion([a, b], coefficient=1.0)
    # For a and b, dx = 0-1 = -1, dist_sq = 1, factor = (1*2*2)/1 = 4.
    # Expected: a.dx += -4, b.dx += 4.
    assert a.dx == pytest.approx(-4.0)
    assert b.dx == pytest.approx(4.0)


def test_apply_gravity_linear():
    """Test linear gravity."""
    # Test linear gravity via apply_gravity.
    node = layout.Node()
    node.mass = 1.0
    node.x, node.y = 3.0, 4.0  # distance 5
    node.dx = node.dy = 0.0

    layout.apply_gravity([node], gravity=1.0, scaling_ratio=1.0, use_strong_gravity=False)
    # lin_gravity: factor = 1/5 = 0.2; dx -= 3*0.2, dy -= 4*0.2.
    assert node.dx == pytest.approx(-0.6)
    assert node.dy == pytest.approx(-0.8)


def test_apply_gravity_strong():
    """Test strong gravity."""
    # Test strong gravity via apply_gravity.
    node = layout.Node()
    node.mass = 1.0
    node.x, node.y = 3.0, 4.0
    node.dx = node.dy = 0.0

    layout.apply_gravity([node], gravity=1.0, scaling_ratio=1.0, use_strong_gravity=True)
    # strong_gravity: factor = 1*1*1 = 1; so dx -=3, dy -=4.
    assert node.dx == pytest.approx(-3.0)
    assert node.dy == pytest.approx(-4.0)


def test_apply_attraction():
    """Test attraction between two nodes."""
    a = layout.Node()
    b = layout.Node()
    a.mass = b.mass = 1.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 1.0, 0.0
    a.dx = a.dy = b.dx = b.dy = 0.0

    edge = layout.Edge()
    edge.node1 = 0
    edge.node2 = 1
    edge.weight = 1.0
    nodes = [a, b]

    layout.apply_attraction(
        nodes,
        [edge],
        distributed_attraction=False,
        coefficient=1.0,
        edge_weight_influence=1.0,
        lin_log_mode=False,
    )
    # For a and b, dx = 0-1 = -1, factor = -1.0, so a.dx += (-1)*(-1)=1, b.dx -= (-1)*(-1)= -1.
    assert a.dx == pytest.approx(1.0)
    assert b.dx == pytest.approx(-1.0)


def test_adjust_speed_and_apply_forces():
    """Test adjusting speed and applying forces."""
    # Create two nodes with simple forces.
    a = layout.Node()
    b = layout.Node()
    a.mass = b.mass = 1.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 1.0, 0.0

    # Set previous and current forces.
    a.old_dx, a.old_dy = 0.0, 0.0
    a.dx, a.dy = 1.0, 0.0
    b.old_dx, b.old_dy = 0.0, 0.0
    b.dx, b.dy = -1.0, 0.0

    speed = 1.0
    speed_efficiency = 1.0
    jitter_tolerance = 1.0

    result = layout.adjust_speed_and_apply_forces([a, b], speed, speed_efficiency, jitter_tolerance)
    # Check that positions have updated (they should not remain the same).
    assert (a.x, a.y) != (0.0, 0.0)
    assert (b.x, b.y) != (1.0, 0.0)
    # Check that returned dictionary contains the keys.
    assert "speed" in result
    assert "speed_efficiency" in result


###############################################################################
# 2. Tests for the Region class (Barnes-Hut tree)
###############################################################################


def test_region_update_mass_and_geometry():
    """Test updating mass and geometry of a region."""
    a = layout.Node()
    b = layout.Node()
    a.mass = 2.0
    b.mass = 4.0
    a.x, a.y = 0.0, 0.0
    b.x, b.y = 4.0, 0.0

    region = layout.Region([a, b])
    # Total mass should be 6.
    assert region.mass == pytest.approx(6.0)
    # Center of mass: ((0*2 + 4*4)/6, 0) = (16/6, 0) = (8/3, 0).
    assert region.mass_center_x == pytest.approx(8 / 3)
    assert region.mass_center_y == pytest.approx(0.0)
    # The size should be roughly twice the max distance from the center.
    expected_size = 2 * (8 / 3)
    assert region.size == pytest.approx(expected_size, rel=0.1)


def test_region_build_subregions():
    """Test building subregions for a region."""
    # Create 4 nodes arranged in different quadrants.
    nodes = []
    positions = [(1, 3), (1, 1), (3, 3), (3, 1)]
    for pos in positions:
        n = layout.Node()
        n.mass = 1.0
        n.x, n.y = pos
        nodes.append(n)
    region = layout.Region(nodes)
    region.build_subregions()
    # Expect up to 4 subregions.
    assert len(region.subregions) == 4
    for sub in region.subregions:
        assert len(sub.nodes) > 0


def test_region_apply_force():
    """Test applying force from a region to a node."""
    # Test that a region with a single node applies the same force as lin_repulsion.
    node = layout.Node()
    node.mass = 1.0
    node.x, node.y = 5.0, 5.0
    node.dx = node.dy = 0.0

    dummy = layout.Node()
    dummy.mass = 1.0
    dummy.x, dummy.y = 1.0, 1.0
    region = layout.Region([dummy])
    # For a region with one node, apply_force calls lin_repulsion.
    region.apply_force(node, theta=100, coefficient=1.0)
    # dx = 5-1 = 4; dist_sq = 32; factor = 1/32 = 0.03125; expected force = 4*0.03125 = 0.125.
    assert node.dx == pytest.approx(0.125)
    assert node.dy == pytest.approx(0.125)

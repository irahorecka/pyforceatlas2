"""
tests/test_pyforceatlas2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pytest

from pyforceatlas2 import pyforceatlas2 as paf2

###############################################################################
# 1. Tests for the ForceAtlas2 class (pyforceatlas2.py)
###############################################################################


def test_forceatlas2_init_numpy():
    """Test the initialization of the ForceAtlas2 class with a simple numpy adjacency matrix."""
    # Create a simple symmetric 3x3 numpy adjacency matrix.
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    forceatlas = paf2.ForceAtlas2(verbose=False)
    nodes, edges = forceatlas.init(matrix, pos=None)
    assert len(nodes) == 3
    for node in nodes:
        # Mass should be degree+1; so minimum is 1.
        assert node.mass >= 1
    # Only upper-triangle edges are taken, so expect 2 edges.
    assert len(edges) == 2


def test_forceatlas2_layout_numpy():
    """Test the ForceAtlas2 layout on a simple 3x3 numpy adjacency matrix."""
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    forceatlas = paf2.ForceAtlas2(verbose=False)
    positions = forceatlas.forceatlas2(matrix, pos=None, iterations=10)
    # positions should be a list with 3 (x,y) tuples.
    assert len(positions) == 3
    for pos in positions:
        assert isinstance(pos, tuple)
        assert len(pos) == 2


def test_forceatlas2_networkx_layout():
    """Test the ForceAtlas2 layout on a simple path graph."""
    G = nx.path_graph(4)
    forceatlas = paf2.ForceAtlas2(verbose=False)
    positions = forceatlas.forceatlas2_networkx_layout(G, pos=None, iterations=10)
    # positions should be a dict mapping each node to a (x,y) tuple.
    assert isinstance(positions, dict)
    assert len(positions) == 4
    for pos in positions.values():
        assert isinstance(pos, tuple)
        assert len(pos) == 2


def test_forceatlas2_init_invalid():
    """Test that passing an invalid adjacency matrix raises a ValueError."""
    # Passing an invalid 'pos' type should raise an assertion.
    forceatlas = paf2.ForceAtlas2(verbose=False)
    with pytest.raises(AssertionError):
        matrix = np.array([[0, 1], [1, 0], [0, 1]], dtype=float)
        forceatlas.init(matrix, pos="invalid")


@pytest.mark.skipif("not __import__('igraph')")
def test_forceatlas2_igraph_layout():
    """Test the ForceAtlas2 layout on an igraph graph."""
    import igraph

    # Create a simple igraph graph with 3 nodes and 2 edges.
    g = igraph.Graph()
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2)])
    forceatlas = paf2.ForceAtlas2(verbose=False)
    layout_obj = forceatlas.forceatlas2_igraph_layout(g, pos=None, iterations=10)
    # layout_obj should be an igraph.Layout and have 3 positions.
    assert isinstance(layout_obj, igraph.Layout)
    assert len(layout_obj) == 3


###############################################################################
# 2. Additional tests for user input and API consistency
###############################################################################


def test_forceatlas2_user_input_networkx_custom():
    """Create a complete NetworkX graph of 5 nodes. Provide a custom initial position dictionary.
    Verify that the returned layout is a dict mapping nodes to (x,y) tuples.
    """
    G = nx.complete_graph(5)
    pos = {i: (float(i), float(i)) for i in G.nodes()}
    forceatlas = paf2.ForceAtlas2(
        outbound_attraction_distribution=True,
        edge_weight_influence=0.5,
        lin_log_mode=True,
        jitter_tolerance=5.0,
        barnes_hut_optimize=True,
        barnes_hut_theta=1.5,
        scaling_ratio=1.0,
        strong_gravity_mode=True,
        gravity=0.5,
        verbose=False,
    )
    new_pos = forceatlas.forceatlas2_networkx_layout(G, pos=pos, iterations=20)

    # Verify output is a dict with five nodes, each with an (x,y) tuple.
    assert isinstance(new_pos, dict)
    assert len(new_pos) == 5
    for key, value in new_pos.items():
        assert isinstance(value, tuple)
        assert len(value) == 2


def test_forceatlas2_user_input_invalid_graph():
    """Passing a non-NetworkX graph should raise a ValueError."""
    forceatlas = paf2.ForceAtlas2(verbose=False)
    with pytest.raises(ValueError):
        forceatlas.forceatlas2_networkx_layout("not a NetworkX graph", pos=None, iterations=10)


def test_forceatlas2_user_input_invalid_positions():
    """Passing an invalid type for initial positions (e.g., a string)
    should raise an AssertionError.
    """
    G = nx.complete_graph(5)
    forceatlas = paf2.ForceAtlas2(verbose=False)
    with pytest.raises(AssertionError):
        forceatlas.forceatlas2_networkx_layout(G, pos="invalid", iterations=10)


def test_forceatlas2_api_consistency():
    """Run the ForceAtlas2 layout on a simple path graph twice:
    once with random initial positions and once with user-provided positions.
    Verify that both return valid dictionaries mapping each node to a (x, y) tuple.
    """
    G = nx.path_graph(10)
    forceatlas = paf2.ForceAtlas2(verbose=False)
    # Layout with random initial positions.
    pos1 = forceatlas.forceatlas2_networkx_layout(G, pos=None, iterations=10)
    # Layout with user-defined positions.
    pos_initial = {n: (0.1 * n, 0.1 * n) for n in G.nodes()}
    pos2 = forceatlas.forceatlas2_networkx_layout(G, pos=pos_initial, iterations=10)

    # Both outputs must be dictionaries with 10 items and valid (x,y) tuples.
    assert isinstance(pos1, dict)
    assert isinstance(pos2, dict)
    assert set(pos1.keys()) == set(pos2.keys())
    for p in pos1.values():
        assert isinstance(p, tuple) and len(p) == 2
    for p in pos2.values():
        assert isinstance(p, tuple) and len(p) == 2

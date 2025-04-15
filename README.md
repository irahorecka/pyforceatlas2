# PyForceAtlas2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pyforceatlas2.svg)](https://pypi.python.org/pypi/pyforceatlas2)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-purple.svg)](https://raw.githubusercontent.com/username/pyforceatlas2/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1371/journal.pone.0098679-blue)](https://doi.org/10.1371/journal.pone.0098679)
[![tests](https://github.com/irahorecka/pyforceatlas2/workflows/tests/badge.svg)](https://github.com/irahorecka/pyforceatlas2/actions)

PyForceAtlas2 is a Python implementation of the [ForceAtlas2](https://doi.org/10.1371/journal.pone.0098679) graph layout algorithm. Originally designed for Gephi, this implementation is optimized for reproducible and high-performance network visualization in Python, supporting both NetworkX and igraph.

## Features

- **High Performance:** Barnes-Hut optimization for large networks.
- **Customizable:** Fine-tune attraction, repulsion, gravity, and speed parameters.
- **Versatile:** Seamless integration with NetworkX and igraph.
- **Reproducible:** Easily set random seeds for deterministic layouts.

## Installation

Install PyForceAtlas2 via pip:

```bash
pip install pyforceatlas2
```

## Usage Example

The following example demonstrates how to compute a layout for a random geometric graph and visualize it with NetworkX and Matplotlib:

```python
"""
Example: ForceAtlas2 Layout on a Random Geometric Graph
--------------------------------------------------------

This script generates a random geometric graph with 400 nodes,
computes a ForceAtlas2 layout, and visualizes the result.
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from pyforceatlas2 import ForceAtlas2

# Set a random seed for reproducibility.
random.seed(888)

# Generate a random geometric graph: nodes are connected if they are within radius 0.2.
G = nx.random_geometric_graph(400, 0.2)

# Initialize ForceAtlas2 with custom parameters.
forceatlas2 = ForceAtlas2(
    outbound_attraction_distribution=False,  # Do not dissuade hubs.
    edge_weight_influence=1.0,
    lin_log_mode=True,                       # Use LinLog mode for attraction forces.
    jitter_tolerance=10.0,                     # Tolerance for oscillations.
    barnes_hut_optimize=True,                  # Enable Barnes-Hut optimization.
    barnes_hut_theta=2.2,                      # Trade-off between accuracy and speed.
    scaling_ratio=2.0,                         # Controls overall repulsion strength.
    strong_gravity_mode=False,                 # Use standard gravity.
    gravity=1.0,                               # Gravitational constant.
    verbose=True,
)

# Compute the layout. Returns a dict mapping each node to an (x, y) coordinate.
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100)

# Visualize the graph.
nx.draw_networkx_nodes(G, positions, node_size=20, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)
plt.axis("off")
plt.show()
```

## API Overview

The main entry point is the ForceAtlas2 class:

```python
from pyforceatlas2 import ForceAtlas2

# Initialize with desired parameters.
forceatlas2 = ForceAtlas2(
    outbound_attraction_distribution=False,
    edge_weight_influence=1.0,
    lin_log_mode=True,
    jitter_tolerance=10.0,
    barnes_hut_optimize=True,
    barnes_hut_theta=2.2,
    scaling_ratio=2.0,
    strong_gravity_mode=False,
    gravity=1.0,
    verbose=True,
)

# For a NetworkX graph:
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100)

# For an igraph graph:
layout_obj = forceatlas2.forceatlas2_igraph_layout(igraph_graph, pos=None, iterations=100)
```

## References

- **ForceAtlas2 Paper:**  
  Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).  
  *ForceAtlas2, a continuous graph layout algorithm for handy network visualization designed for the Gephi software.*  
  PLoS ONE, 9(6), e98679.  
  [DOI: 10.1371/journal.pone.0098679](https://doi.org/10.1371/journal.pone.0098679)

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/irahorecka/pyforceatlas2).

## License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html). See the [LICENSE](LICENSE) file for details.

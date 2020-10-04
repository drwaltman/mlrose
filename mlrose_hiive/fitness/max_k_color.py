""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class MaxKColor:
    """Fitness function for Max-k color optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the color of node i, as the number of pairs of adjacent nodes
    of the same color.

    Parameters
    ----------
    edges: list of pairs
        List of all pairs of connected nodes. Order does not matter, so (a, b)
        and (b, a) are considered to be the same.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The MaxKColor fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self, edges, maximize=False):

        # Remove any duplicates from list
        edges = list({tuple(sorted(edge)) for edge in edges})

        self.graph_edges = None
        self.edges = edges
        self.maximize = maximize
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0

        # this is the count of neigbor nodes with the same state value.
        # Therefore state value represents color.
        # This is NOT what the docs above say.

        if self.graph_edges is not None:
            if self.maximize:
                fitness = sum(int(state[n1] != state[n2]) for (n1, n2) in self.graph_edges)
            else:
                fitness = sum(int(state[n1] == state[n2]) for (n1, n2) in self.graph_edges)
        else:
            fitness = 0
            for (n1, n2) in self.edges:
                if self.maximize:
                    # Check for adjacent nodes of different color
                    if state[n1] != state[n2]:
                        fitness += 1
                else:
                    # Check for adjacent nodes of the same color
                    if state[n1] == state[n2]:
                        fitness += 1

        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type

    def set_graph(self, graph):
        self.graph_edges = [e for e in graph.edges()]

"""
Just functions that are used in multiple places just to read/write instances
"""

def create_tsp_problem_object(G, cost="weight"):
    # Assert that the graph is complete and undirected, othewrise the TSP problem is not well defined.
    assert G.is_directed() == False, "The graph must be undirected."
    n = G.number_of_nodes()
    assert G.number_of_edges() == n * (n - 1) // 2, "The graph must be complete."

    # Nice!
    problem = tsplib95.models.StandardProblem()
    problem.name = "tmp"
    problem.type = "TSP"
    problem.dimension = G.number_of_nodes()
    problem.edge_weight_type = "EXPLICIT"
    problem.edge_weight_format = "UPPER_ROW"
    edge_weight_section = []
    for i in range(problem.dimension):
        for j in range(i + 1, problem.dimension):
            edge_weight_section.append(int(G[i][j][cost]))
    problem.edge_weights = [edge_weight_section]
    return problem

def write_tsplib(G, filename, cost="weight"):
    """
    Write a TSPLIB file from a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to write.
    filename : str
        The name of the file to write.
    cost : str
        The edge attribute to use as cost. Default: 'weight'
    """
    # TODO incomplete?
    problem = create_tsp_problem_object(G, cost=cost)
    with open(filename, "w") as f:
        problem_str = str(problem).replace("EDGE_WEIGHT_SECTION:", "EDGE_WEIGHT_SECTION:\n")
        f.write(problem_str)

def read_tsplib(filename):
    """
    Read a TSPLIB file and return a NetworkX graph.

    Note: done because tsplib95 might not always work.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Returns
    -------
    G : networkx.Graph
        The graph represented by the TSPLIB file.
    """
    # TODO
    pass

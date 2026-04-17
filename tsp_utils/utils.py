"""
Just functions that are used in multiple places just to read/write instances
"""
import tsplib95
import networkx as nx

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

def read_upperrow(lines, n):
    i = 0
    j = 1
    cont = 0
    out = dict(zip([(i, j) for i in range(n) for j in range(i + 1, n)], [0] * (n * (n - 1) // 2)))
    for line in lines:
        if line.strip() == "EOF":
            break
        else:
            line_vec = line.strip().split(" ")
            for w in line_vec:
                if w != "":
                    out[(i, j)] = int(w)
                    cont += 1
                    if j == n - 1:
                        i += 1
                        j = i + 1
                    else:
                        j += 1
    # Sanity checks
    assert cont == n * (n - 1) // 2, f"Expected {n * (n - 1) // 2} edge weights, but got {cont}."

    return out



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
    with open(filename, "r") as f:
        lines = f.readlines()

    G = None # Will be either graph or digraph, depending on the type of problem
    n = None
    for line in lines:
        if line.startswith("TYPE:"):
            problem_type = line.split(":")[1].strip()
            if problem_type == "TSP":
                G = nx.Graph()
                G.graph["name"] = filename
            elif problem_type == "ATSP":
                G = nx.DiGraph()
                G.graph["name"] = filename
            else:
                raise ValueError(f"Unsupported problem type: {problem_type}")
        elif line.startswith("DIMENSION:"):
            n = int(line.split(":")[1].strip())
            G.add_nodes_from(range(n))
        elif line.startswith("EDGE_WEIGHT_FORMAT:"):
            type = line.split(":")[1].strip() # Not used, but we assume it's always UPPER_ROW
            if type != "UPPER_ROW":
                raise NotImplementedError(f"Unsupported edge weight format: {type}")
            else:
                # Save the idx of the line where the edge weights start
                # So the idea is that, after the edge weight type, you have new line, edge weight section, new line, hence + 2
                edge_weight_start_idx = lines.index(line) + 2
                edges = read_upperrow(lines[edge_weight_start_idx:], n)
                for (i, j), w in edges.items():
                    G.add_edge(i, j, weight=w)
                break

    # So far, have you read something meaningful?
    assert G is not None, "Could not find the problem type in the file."
    assert n is not None, "Could not find the dimension in the file."

    # Pick a random edge, the weight should be there
    assert "weight" in G[0][1], "Could not find the edge weights in the file."

    return G
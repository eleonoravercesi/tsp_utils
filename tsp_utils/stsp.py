import networkx
from pyscipopt import Model, quicksum

def delta(S, edges):
    """
    Returns the edges having exactly one endpoint in S

    Parameters
    ----------
    S : set
        Set of nodes in the graph
    edges : list
        List of edges in the graph

    Returns
    -------
    list
        List of edges having exactly one endpoint in S
    """
    return [e for e in edges if (e[0] in S) != (e[1] in S)]


def solve_tsp(G, cost="weight", verbose=False):
    """
    Adapted from https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/tsp.py
    Parameters
    ----------
    G : networkx.Graph
        NetworkX graph
    cost : string
        Cost function, must be one of the edges attributes. Default: 'weight'
    verbose : bool
        Whether to print verbose output. Default: False

    Returns:
    --------
    TODO
    """


    def addcut(cut_edges):
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False
        model.freeTransform()
        for S in Components:
            T = set(V) - set(S)
            model.addCons(quicksum(x[i, j] for i in S for j in T if j > i) +
                          quicksum(x[i, j] for i in T for j in S if j > i) >= 2)
        return True

    # main part of the solution process:
    model = Model()

    if not verbose:
        model.hideOutput()  # silent/verbose mode

    n = G.number_of_nodes()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    V = range(n)

    x = {}
    for e in edges:
        i, j = e
        x[i, j] = model.addVar(vtype="C", lb=0, ub=1, name="x(%s,%s)" % (i, j))

    for i in V:
        model.addCons(quicksum(x[u, v] for (u, v) in delta({i}, edges)) == 2, "degree(%s)" % i)

    model.setObjective(quicksum(G[e[0]][e[1]][cost] * x[e[0], e[1]] for e in edges), "minimize")

    EPS = 1.e-6
    isMIP = False
    while True:
        model.optimize()
        non_zero_edges = []
        for (i, j) in x:
            if model.getVal(x[i, j]) > EPS:
                non_zero_edges.append((i, j))

        if addcut(non_zero_edges) == False:
            if isMIP:  # integer variables, components connected: solution found
                break
            model.freeTransform()
            for (i, j) in x:  # all components connected, switch to integer model
                model.chgVarType(x[i, j], "B")
                isMIP = True

    # Get the runtime
    runtime = model.getTotalTime()

    return model.getObjVal(), non_zero_edges, runtime


def solve_tsp_fixed_edge(G, e=None, cost="weight", verbose=False):
    """
    Adapted from https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/tsp.py

    Solve the TSP by forcing an edge beeing in the solution

    Parameters
    ----------
    G : networkx.Graph
        NetworkX graph
    cost : string
        Cost function, must be one of the edges attributes. Default: 'weight'
    verbose : bool
        Whether to print verbose output. Default: False

    Returns:
    --------
    opt: float
        Optimal value of the objective function
    edges: list
        List of edges in the optimal solution
    """
    if e is None:
        raise ValueError("Edge must be provided")

    def addcut(cut_edges):
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False
        model.freeTransform()
        for S in Components:
            T = set(V) - set(S)
            model.addCons(quicksum(x[i, j] for i in S for j in T if j > i) +
                          quicksum(x[i, j] for i in T for j in S if j > i) >= 2)
        return True

    # main part of the solution process:
    model = Model()

    if not verbose:
        model.hideOutput()  # silent/verbose mode

    n = G.number_of_nodes()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    V = range(n)

    x = {}
    for e in edges:
        i, j = e
        x[i, j] = model.addVar(vtype="C", lb=0, ub=1, name="x(%s,%s)" % (i, j))

    # Add the fixed edge constrant
    i, j = e
    model.addCons(x[i, j] == 1, "fixed_edge(%s,%s)" % (i, j))

    for i in V:
        model.addCons(quicksum(x[u, v] for (u, v) in delta({i}, edges)) == 2, "degree(%s)" % i)

    model.setObjective(quicksum(G[e[0]][e[1]][cost] * x[e[0], e[1]] for e in edges), "minimize")

    EPS = 1.e-6
    isMIP = False
    while True:
        model.optimize()
        non_zero_edges = []
        for (i, j) in x:
            if model.getVal(x[i, j]) > EPS:
                non_zero_edges.append((i, j))

        if addcut(non_zero_edges) == False:
            if isMIP:  # integer variables, components connected: solution found
                break
            model.freeTransform()
            for (i, j) in x:  # all components connected, switch to integer model
                model.chgVarType(x[i, j], "B")
                isMIP = True

    # Return the runtime
    runtime = model.getTotalTime()


    return model.getObjVal(), non_zero_edges, runtime
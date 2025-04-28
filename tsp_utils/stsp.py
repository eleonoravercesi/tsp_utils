import networkx
from pyscipopt import Model, quicksum
import pathlib
import tsplib95
import os
import subprocess

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

def create_tsp_problem_object(G, cost="weight"):
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

def parse_log(filename):
    F = open(filename)
    lines = F.readlines()
    F.close()
    ot_line = next(filter(lambda x : "Optimal Solution:" in x, list(lines.__reversed__())))
    bb_nodes_line = ""
    try:
        bb_nodes_line = next(filter(lambda x : "Number of bbnodes:" in x, list(lines.__reversed__())))
    except:
        pass
    time_line = next(filter(lambda x : "Total Running Time:" in x , list(lines.__reversed__())))
    ot = float(ot_line.split(":")[1].strip())
    bb_nodes = 0
    if len(bb_nodes_line) >= 1:
        bb_nodes = float(bb_nodes_line.split(":")[1].strip())
    time = float(time_line.split(":")[1].split("(")[0])
    return ot, bb_nodes, time

def run_concorde(G, cost="weight", concorde_path=None, seed = None, options=[], verbose=False, remove_all=False):
    """
    Run the Concorde TSP solver on a given graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to solve.
    concorde_path : str
    """
    assert concorde_path != None, "Concorde path must be provided"

    concorde_path = pathlib.Path(concorde_path)

    # Create a directory named tmp to store temporary files
    tmp_dir = "./tmp/"
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    problem = create_tsp_problem_object(G, cost=cost)
    tsp_file = tmp_dir + "tmp.tsp"
    with open(tsp_file, 'w') as f:
        problem.write(f)

    # Change the directory
    os.chdir(tmp_dir)

    # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
    # Run Concorde
    # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

    # is verbose?
    if not verbose:
        options += ["-x"]
    # Do we want to use the seed?
    if seed != None:
        options += ["-s {}".format(seed)]

    # Merge all the options together
    options = " ".join(options)

    # ~/libraries/concorde/build/TSP/concorde tmp.tsp > tmp.log
    subprocess.run(f"{concorde_path} {options} tmp.tsp > tmp.log", shell=True)
    ot, bb_nodes, time = parse_log("tmp.log")

    # Return to the original directory
    os.chdir("..")

    if remove_all:
        # Remove the tmp directory
        os.rmdir(tmp_dir, recursive=True)
    return ot, bb_nodes, time

def run_concorde_LB(G, cost="weight", concorde_path=None, seed = None, options=[], verbose=False, remove_all=False):
    # is verbose?
    if not verbose:
        options += ["-x"]
    # Do we want to use the seed?
    if seed != None:
        options += ["-s {}".format(seed)]

    # Just the LB
    options += ["-I"]

    # Merge all the options together
    options = " ".join(options)
    return run_concorde(G, cost=cost, concorde_path=concorde_path, options=options, seed=seed, verbose=verbose, remove_all=remove_all)







#
# def run_concorde_LB(tsp_file, root, logname="test.log", remove=True, change_dir=None):
#     try:
#         assert root[-1] == "/"
#     except:
#         root = root + "/"
#     assert type(change_dir) == str or change_dir is None, "change_dir must be a string path of the directory you want to run things in or None"
#     # If you have to change dir for the run
#     # Check if the directory RUN
#     if change_dir is not None:
#         if not os.path.exists(change_dir):
#             os.mkdir(change_dir)
#         # Change directory
#         os.chdir(change_dir)
#     subprocess.run("{}{} -x -I ../{} > {}".format(root, "concorde", tsp_file, logname), shell=True)
#     # Open the log file
#     F = open(logname, "r")
#     lines = F.readlines()
#     F.close()
#
#     # Get the line with written "Bound: "
#     lb_line = next(filter(lambda x : "Bound: " in x, lines))
#     lb = float(lb_line.split(":")[1].split("(")[0])
#     # Get the line with the time
#     time_line = next(filter(lambda x: "Total Running Time:" in x, list(lines.__reversed__())))
#     time = float(time_line.split(":")[1].strip().split("(")[0])
#
#     if remove:
#         os.remove(logname)
#     if change_dir is not None:
#         os.chdir("..")
#     return lb, time
#
#

#

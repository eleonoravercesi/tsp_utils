import networkx as nx
import time
from tsp_utils.stsp import solve_tsp, solve_tsp_fixed_edge, run_concorde
import numpy as np

def test_solve_tsp():
    n = 17
    c_max = 100
    # Create random list of edges
    C = np.random.randint(1, c_max,  (n*(n-1)//2, ))

    C_mat = np.zeros((n, n))

    G = nx.Graph()
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            c_i_j = C[e]
            G.add_edge(i, j, weight=c_i_j)
            C_mat[i, j] = c_i_j
            C_mat[j, i] = c_i_j
            e += 1


    opt, edges, runtime = solve_tsp(G)
    print("MIP runtime: ", runtime)

    # DP
    start = time.time()
    tour, opt_real = solve_tsp_dynamic_programming(C_mat)
    end = time.time()
    print("DP runtime: ", end - start)

    assert opt == opt_real, f"opt: {opt}, opt_real: {opt_real}"

    opt_fe, _, runtime = solve_tsp_fixed_edge(G, (0, 1))
    print("One edge fixed runtime: ", runtime)

    assert opt_fe >= opt_real

if __name__ == "__main__":
    n = 17
    c_max = 100
    # Create random list of edges
    C = np.random.randint(1, c_max, (n * (n - 1) // 2,))

    G = nx.Graph()
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            c_i_j = C[e]
            G.add_edge(i, j, weight=c_i_j)
            e += 1

    print(run_concorde(G, concorde_path="~/libraries/concorde/build/TSP/concorde"))

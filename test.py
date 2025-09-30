#%%
# Test the function dual_sep

from tsp_utils.stsp import dual_sep, solve_tsp, create_tsp_problem_object
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure
import numpy as np

from tsp_utils.tsp import run_LKH

#%% Test LKH3
n = 6
np.random.seed(42)
m = n * ( n - 1) // 2
G = nx.complete_graph(n)
for (u, v) in G.edges():
    G.edges[u, v]['weight'] = np.random.randint(1, 10)

problem = create_tsp_problem_object(G)
with open("tmp.tsp", "w") as f:
    problem_str = str(problem).replace("EDGE_WEIGHT_SECTION:", "EDGE_WEIGHT_SECTION:\n")
    f.write(problem_str)
costs, trials, runtime = run_LKH("tmp.tsp", lkh3_path="/home/vercee/libraries/LKH-3.0.9/LKH")

# #%%
# # Create a path graph on n nodes
# debug = False
# n = 12
# G = nx.path_graph(n)
# # Add weights to the edges
# for (u, v) in G.edges():
#     G.edges[u, v]['weight'] = 1
#
# eps = 0.01
# for i in range(0, n):
#     if i + 2 < n:
#         G.add_edge(i, (i + 2), weight=1 + eps)
#         if debug:
#             print(f"Adding edge {(i, (i + 2))} with weight {1 + eps}")
#
# # Complete the graph
# G = metric_closure(G)
#
# # Solve the TSP
# TSP, edges, runtime = solve_tsp(G, cost="distance")
#
# # Solve the dual
# y_val, d_val, u_val, DTSP = dual_sep(G, cost="distance", write=False)
#
# assert round(DTSP, 5) == round(TSP, 5)
#
# for e in u_val.keys():
#     if u_val[e] > 0:
#         print(e, round(u_val[e],4))
#
# for S in d_val.keys():
#     if d_val[S] > 0:
#         print(S, round(d_val[S],4))
#
# for i in range(n):
#    if y_val[i] > 1e-10:
#        print(i, round(y_val[i], 4))

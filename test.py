#%%
# Test the function dual_sep

from tsp_utils.stsp import dual_sep, solve_tsp
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure

# Create a path graph on n nodes
debug = False
n = 12
G = nx.path_graph(n)
# Add weights to the edges
for (u, v) in G.edges():
    G.edges[u, v]['weight'] = 1

eps = 0.01
for i in range(0, n):
    if i + 2 < n:
        G.add_edge(i, (i + 2), weight=1 + eps)
        if debug:
            print(f"Adding edge {(i, (i + 2))} with weight {1 + eps}")

# Complete the graph
G = metric_closure(G)

# Solve the TSP
TSP, edges, runtime = solve_tsp(G, cost="distance")

# Solve the dual
y_val, d_val, u_val, DTSP = dual_sep(G, cost="distance", write=False)

assert round(DTSP, 5) == round(TSP, 5)

for e in u_val.keys():
    if u_val[e] > 0:
        print(e, round(u_val[e],4))

for S in d_val.keys():
    if d_val[S] > 0:
        print(S, round(d_val[S],4))

for i in range(n):
   if y_val[i] > 1e-10:
       print(i, round(y_val[i], 4))

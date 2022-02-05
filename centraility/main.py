import functools
from math import sqrt
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

def degree_centrality(G):
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree()}
    return centrality

def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    if distance is not None:
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            if wf_improved:
                s = (len(sp) - 1.0) / (len_G - 1)
                _closeness_centrality *= s
        closeness_centrality[n] = _closeness_centrality
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality 

def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    if nstart is None:
        nstart = {v: 1 for v in G}
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = G.number_of_nodes()
    for i in range(max_iter):
        xlast = x
        x = xlast.copy()
        for n in x:
            for nbr in G[n]:
                w = G[n][nbr].get(weight, 1) if weight else 1
                x[nbr] += xlast[n] * w
        norm = sqrt(sum(z ** 2 for z in x.values())) or 1
        x = {k: v / norm for k, v in x.items()}
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


G = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.2, seed=None, directed=False)

DC = degree_centrality(G)
CC = closeness_centrality(G)
BC = nx.algorithms.centrality.betweenness_centrality(G)
EC = eigenvector_centrality(G)
PRC = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)

sorted_keys = sorted(DC, key=DC.get)  
sorted_keys2 = sorted(CC, key=CC.get)  
sorted_keys3 = sorted(BC, key=BC.get)  
sorted_keys4 = sorted(EC, key=EC.get)  
sorted_keys5 = sorted(PRC, key=PRC.get)  

df = pd.DataFrame({'Degree Centrality' : sorted_keys[:10], 'Closeness Centrality' : sorted_keys2[:10], 'Betweenness Centrality' : sorted_keys3[:10], 'Eigenvector Centrality' : sorted_keys4[:10], 'Page Rank Centrality' : sorted_keys5[:10]})
df.to_csv('items.csv', index=False, encoding='utf-8')

plt.plot(sorted_keys[:10], label = "Degree Centrality")
plt.plot(sorted_keys2[:10], label = "Closeness Centrality")
plt.plot(sorted_keys3[:10], label = "Betweenness Centrality")
plt.plot(sorted_keys4[:10], label = "Eigenvector Centrality")
plt.plot(sorted_keys5[:10], label = "Page Rank Centrality")

plt.savefig('fig.png')
plt.legend()
plt.show()

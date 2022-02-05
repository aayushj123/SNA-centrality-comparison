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
CC = nx.algorithms.centrality.closeness_centrality(G)
BC = nx.algorithms.centrality.betweenness_centrality(G)
EC = eigenvector_centrality(G)
PRC = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)




key = list(DC.values())
key2 = list(CC.values())
key3 = list(BC.values())
key4 = list(EC.values())
key5 = list(PRC.values())

plt.plot(key, label = "Degree Centrality")
plt.plot(key2, label = "Closeness Centrality")
plt.plot(key3, label = "Betweenness Centrality")
plt.plot(key4, label = "Eigenvector Centrality")
plt.plot(key5, label = "Page Rank Centrality")
plt.legend()
plt.savefig('cen.png')
plt.show()
plt.ylabel("Centrality")
plt.xlabel("Number of nodes")    
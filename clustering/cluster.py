import random
from matplotlib import cm, pyplot as plt
import networkx as nx

G = nx.generators.random_graphs.erdos_renyi_graph(1000, 0.2, seed=None, directed=False)
nx.draw(G)
plt.savefig("Graph.png", format="PNG")

def average_clustering(G, trials=1000):
	
	n = len(G)
	triangles = 0
	nodes = G.nodes()
	for i in [int(random.random() * n) for i in range(trials)]:
		nbrs = list(G[nodes[i]])
		if len(nbrs) < 2:
			continue
		u, v = random.sample(nbrs, 2)
		if u in G[v]:
			triangles += 1
	return triangles / float(trials)

cc=nx.average_clustering(G)
c=nx.clustering(G)

gc=G.subgraph(max(nx.connected_components(G)))
lcc=nx.clustering(gc)
cmap=plt.get_cmap('autumn')
norm=plt.Normalize(0,max(lcc.values()))
node_colors=[cmap(norm(lcc[node])) for node in gc.nodes]
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
nx.draw_spring(gc, node_color=node_colors, with_labels=True, ax=ax1)
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), label='Clustering', shrink=0.95, ax=ax1)

ax2.hist(lcc.values(), bins=10)
ax2.set_xlabel('Clustering')
ax2.set_ylabel('Frequency')

plt.savefig('cls.png')
plt.tight_layout()
plt.show()

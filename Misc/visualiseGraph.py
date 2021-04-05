import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# graph = np.load("graphs/GBSG_graph.npy")
graph = np.load("graphs/kkbox_graph.npy")
G = nx.from_numpy_matrix(graph, create_using = nx.DiGraph)

# node_map = {0: "time", 1: "hthreat", 2: "grade", 3: "menostat", 4: "age", 5: "posnodal", 6: "prm", 7: "esm"}
node_map = {0: "time", 1: "x1", 2: "x2", 3: "x3", 4: "x4", 5: "x5", 6: "x6", 7: "x7", 8: "x8", 9: "x9", 10: "x10", 11: "x11", 12: "x12", 13: "x13", 14: "x14", 15: "x15", 16: "x16"}
#CA4790
G = nx.relabel_nodes(G, node_map)
random_pos = nx.random_layout(G, seed=1189)
# plt.figure(1,figsize=(8,4)) 
nx.draw_circular(G, node_color = '#C7304e', with_labels = True, node_shape = 'o', edge_color = 'black', width = 1, node_size = 2000,  alpha = 1, linewidths = 0.5, font_size=15, font_color = 'white', arrowsize = 10)
plt.savefig("Visualised_graph_kkbox.png")
plt.show()
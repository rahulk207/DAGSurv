import pickle
import numpy as np
import networkx as nx

f = open("digraph_3.p", 'rb')
a = pickle.load(f)

m = nx.adjacency_matrix(a)
np.save("synthetic_graph_3", m.todense())
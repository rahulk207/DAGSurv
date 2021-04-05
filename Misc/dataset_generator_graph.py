import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
import csv
import networkx as nx
import pandas as pd
import pickle

np.random.seed(0)
def generate_DAG(d, degree):
    w_range = (0.5, 2.0)
    prob = float(degree) / (d - 1)
    B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    print(W)
    G = nx.DiGraph(W)
    return G

def simulate_sem(G: nx.DiGraph,
                 n: int,
                 sem_type = "linear-exp",
                 linear_type = "nonlinear_1",
                 noise_scale = 1.0):
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        data: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    data = np.zeros([n, d])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = data[:, parents].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(data[:, parents] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (data[:, parents]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                if(j==0):
                    data[:, j] = np.maximum(np.zeros(n), eta + np.random.normal(loc=200, scale=100, size=n))
                else:
                    data[:, j] = eta + np.random.normal(loc=0, scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                data[:, j] = eta + np.random.normal(loc=0, scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                data[:, j] = 2.*np.sin(eta) + eta + np.random.normal(loc=0, scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            if(j==0):
                data[:, j] = np.maximum(np.zeros(n), 90*np.exp(eta) + np.random.normal(loc=30, scale=noise_scale+69, size=n))
            else:
                data[:, j] = eta + np.random.normal(loc=0, scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            data[:, j] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    return data

num_nodes = 30
each_node_degree = 6
num_samples = 10000
labels=[]

G = generate_DAG(num_nodes, each_node_degree)
sem = simulate_sem(G, num_samples)

max_t = np.max(sem[:][0])
print(sem[:][0])

for i in range(num_samples):
    r=np.random.randint(2)
    labels.append([r])
    if r == 0:
        sem[i][0] = np.random.uniform(0, max_t+1)

with open("digraph_3.p", 'wb') as f:
    pickle.dump(G, f)

with open("semx_3.p", 'wb') as f:
    pickle.dump(sem, f)

data = np.append(sem, labels, axis=1)

data = pd.DataFrame(data.tolist(), columns=['time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'label'])
data.to_csv("synthetic_final_3.csv")

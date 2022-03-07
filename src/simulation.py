import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Setup graph
def sim_data(A, n, Sigma=0.1, plot=False, seed=None):
    burnin = 0
    data = sim_data_inner(A, n+burnin, Sigma=Sigma, plot=plot, seed=seed)
    return data[burnin:]


def sim_data_inner(A, n, Sigma=0.1, plot=False, seed=None):

    # Plot structure of A for sanity check
    if plot:
        G = nx.from_numpy_matrix(A.T, create_using=nx.DiGraph)
        nx.draw_circular(G, with_labels=True)
        plt.show()

    rng = np.random.RandomState(seed=seed)
    N = lambda sigma, n: rng.multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma, size=n)

    # Initialize dimension
    d = A[1].shape[1]
    data = np.zeros((n, d))

    if isinstance(Sigma, float):
        Sigma = Sigma*np.eye(d)

    # Remove versions of A that are all 0
    A = {k: v for (k,v) in A.items() if np.any(v)}
    max_lags = max(list(A.keys()))

    # Fill gaps, i.e. if only lag 1 and 3 are present, make lag 2
    for j in range(1, max_lags + 1):
        if not j in A.keys(): A[j] = np.zeros((d, d))
    A_lags = np.concatenate([A[j] for j in range(1, max_lags+1)], axis=1)

    # Initialize first steps
    for i in range(max_lags):
        data[i] = N(sigma=Sigma, n=1)

    # Loop
    noise = N(sigma=Sigma, n=n)
    # was range(1, n-1) before, imho should be as follows:
    for i in range(0, n-1):
        data[i+1] = A_lags@np.concatenate(
            [data[i-j] for j in range(max_lags)]) + noise[i]

    return data


def get_skeleton(id_I, id_X, id_H, id_Y):
    d = len(id_I)+len(id_X)+len(id_H)+len(id_Y)
    A_empty = np.zeros(shape=(d,d))

    for id in [id_I, id_X, id_H, id_Y]:
        for i in id:
            for j in id:
                A_empty[i,j] = 1

    for id_1, id_2 in [(id_I, id_X), (id_X, id_Y), (id_H, id_X), (id_H, id_Y)]:
        for i in id_1:
            for j in id_2:
                A_empty[j, i] = 1
    return A_empty

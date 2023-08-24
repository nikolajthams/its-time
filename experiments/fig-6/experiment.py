from tqdm import tqdm
from src import simulation, var_iv
import pandas as pd
import numpy as np
MSE = lambda x: (x**2).sum()
from multiprocessing import Pool, cpu_count

# Set dimensions
dI = 1
dX = 2
dH = 1
dY = 1
d = dI + dX + dH + dY

tol = 0.1

# Get indices for data
id_I, id_X, id_H, id_Y = np.split(np.arange(d), np.cumsum([dI, dX, dH, dY]))[:4]

# Get skeleton of matrix A
skel = simulation.get_skeleton(id_I, id_X, id_H, id_Y)

# Update skeleton to have diagonal X-part
skel[dI:(dI+dX), dI:(dI+dX)] = np.eye(2)

results = []

def sample_alphas(skel):
    return skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)


# for rep_mat in tqdm(range(100)):
def one_exp(i=None):
    out = []
    for delta in [0, 0.01, 0.1, 0.5, 1]:
        # Simulate parameters until stable system
        # b = 1*(np.random.uniform(size=skel.shape) < 0.5)
        # A1 = skel*(b*np.random.uniform(-0.99, -0.5, size=skel.shape) + (1-b)*np.random.uniform(0.5, 0.99, size=skel.shape))
        A1 = sample_alphas(skel)
        # A1 = np.random.uniform(-0.99, 0.99, size=skel.shape)*skel
        A1[dI:(dI+dX), dI:(dI+dX)] = np.diag([-0.6, -0.6+delta])
        while np.max(abs(np.linalg.eigvals(A1)))>1:
            # b = 1*(np.random.uniform(size=skel.shape) < 0.5)
            # A1 = skel*(b*np.random.uniform(-0.99, -0.5, size=skel.shape) + (1-b)*np.random.uniform(0.5, 0.99, size=skel.shape))
            # A1 = np.random.uniform(-0.99, 0.99, size=skel.shape)*skel
            A1 = sample_alphas(skel)
            A1[dI:(dI+dX), dI:(dI+dX)] = np.diag([-0.6, -0.6+delta])

        # Save parameters at lag 1 (gives VAR(1) process)
        A = {1: A1}

        for n in [100, 500, 1000, 5000, 10000, 50000]:
            error_niv = []
            try:
                for rep in range(1):
                    # Simulate data
                    data = simulation.sim_data(A, n=n, Sigma=0.1)
                    I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

                    beta_0 = A[1][id_Y][:,id_X].T

                    # Compute estimators
                    error_niv.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y) - beta_0))

                out.append({"delta": delta, "n": n, "method": "TS-NIV", "error": np.mean(error_niv)})
            except: 
                print("An error occurred")
    return out

if __name__ == '__main__':
    n_reps = 1000
    results = list(tqdm(Pool(cpu_count()-1).imap_unordered(one_exp, range(n_reps)), total=n_reps))

    df = pd.DataFrame([i for res in results for i in res])
    df.to_csv("experiments/fig-6/results.csv", index=False)

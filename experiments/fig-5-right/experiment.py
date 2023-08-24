from tqdm import tqdm
from src import simulation, var_iv
import pandas as pd
import numpy as np
MSE = lambda x: (x**2).sum()

# Set dimensions
dI = 3
dX = 2
dH = 1
dY = 1
d = dI + dX + dH + dY

tol = 0.1

# Get indices for data
id_I, id_X, id_H, id_Y = np.split(np.arange(d), np.cumsum([dI, dX, dH, dY]))[:4]

# Get skeleton of matrix A
skel = simulation.get_skeleton(id_I, id_X, id_H, id_Y)

# Update skeleton to have independent instruments
skel[:dI, :dI] = np.eye(dI)

def sample_alphas(skel):
    return skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)

results = []

for rep_mat in tqdm(range(1000)):
    for n in [int(n) for n in [1000]]:
        # Simulate parameters until stable system
        # A1 = skel*np.random.uniform(-0.99, 0.99, size=skel.shape)
        A1 = sample_alphas(skel)
        
        while np.max(abs(np.linalg.eigvals(A1)))>1-tol:
            # A1 = skel*np.random.uniform(-0.99, 0.99, size=skel.shape)
            A1 = sample_alphas(skel)
        # Save parameters at lag 1 (gives VAR(1) process)
        A = {1: A1}

        # Simulate data
        data = simulation.sim_data(A, n=n, Sigma=1.0)
        I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

        beta_0 = A[1][id_Y][:,id_X].T

        niv_single, niv_all = [], []
        for rep in range(10):
            # Compute estimators
            niv_single.append(MSE(var_iv.ts_niv(I=I[:,0], X=X, Y=Y, n_lags=2*dI) - beta_0))
            niv_all.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y, n_lags=2) - beta_0))

        results.append({"n": n, "niv_all": np.mean(niv_all), "niv_single": np.mean(niv_single)})



df = pd.DataFrame(results)
df.to_csv("experiments/fig-5-right/results.csv", index=False)



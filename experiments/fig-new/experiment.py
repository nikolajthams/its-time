from tqdm import tqdm
from src import simulation, var_iv
import pandas as pd
import numpy as np
from multiprocessing import Pool
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

strengths = [0.5**i for i in range(6)]

def sample_alphas(skel, instrument_strength=1):
    out = skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)
    out[id_X][:, id_I] = out[id_X][:, id_I]*instrument_strength
    return out

def one_simulation(i=None):
    results = []
    
    for strength in strengths:
        # Simulate parameters until stable system
        A1 = sample_alphas(skel)
        while np.max(abs(np.linalg.eigvals(A1)))>1-tol:
            A1 = sample_alphas(skel)
        # Save parameters at lag 1 (gives VAR(1) process)
        A = {1: A1}

        for n in map(int, [1e2, 1e3, 1e4]):
            error_civ, error_niv_lags = [], []
            for rep in range(10):
                # Simulate data
                data = simulation.sim_data(A, n=n, Sigma=1.0)
                I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

                beta_0 = A[1][id_Y][:,id_X].T

                # Compute two CIV estimators
                error_civ.append(MSE(var_iv.ts_civ(I=I, X=X, Y=Y) - beta_0))
                
                # Compute two NIV estimators
                error_niv_lags.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y, n_lags=3) - beta_0))


            # Save results
            results.append({"n": n, "method":"CIV$_{I,X,Y}$", "error": np.mean(error_civ), "strength": strength})
            results.append({"n": n, "method":"NIV$_{3 \\textrm{ lags}}$", "error": np.mean(error_niv_lags), "strength": strength})
    
    return results

results = one_simulation(0)

if __name__ == "__main__":
    n_reps = 10
    results = [i for x in tqdm(map(one_simulation, range(n_reps)), total=n_reps) for i in x]
    df = pd.DataFrame(results)
    df.to_csv("experiments/fig-new/results.csv", index=False)

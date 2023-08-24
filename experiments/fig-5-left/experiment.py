from tqdm import tqdm
from src import simulation, var_iv
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
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

def sample_alphas(skel):
    return skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)


def one_simulation(i=None):
    results = []
    # Simulate parameters until stable system
    # A1 = skel*np.random.uniform(-0.99, 0.99, size=skel.shape)
    A1 = sample_alphas(skel)
    while np.max(abs(np.linalg.eigvals(A1)))>1-tol:
        # A1 = skel*np.random.uniform(-0.99, 0.99, size=skel.shape)
        A1 = sample_alphas(skel)
    # Save parameters at lag 1 (gives VAR(1) process)
    A = {1: A1}

    for n in [int(n) for n in [1e2, 1e3, 1e4]]:
        error_civ, error_niv, error_civ_I3, error_niv_lags = [], [], [], []
        for rep in range(10):
            # Simulate data
            data = simulation.sim_data(A, n=n, Sigma=1.0)
            I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

            beta_0 = A[1][id_Y][:,id_X].T

            # Compute two CIV estimators
            error_civ.append(MSE(var_iv.ts_civ(I=I, X=X, Y=Y) - beta_0))
            error_civ_I3.append(MSE(var_iv.ts_civ(I=I, X=X, Y=Y, only_I_as_condition=True) - beta_0))
            
            # Compute two NIV estimators
            error_niv.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y) - beta_0))
            error_niv_lags.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y, n_lags=3) - beta_0))


        # Save results
        results.append({"n": n, "method":"CIV$_{I,X,Y}$", "error": np.mean(error_civ)})
        results.append({"n": n, "method":"CIV$_{I}$", "error": np.mean(error_civ_I3)})
        results.append({"n": n, "method":"NIV$_{1 \\textrm{ lag}}$", "error": np.mean(error_niv)}) #TODO: CHANGED FROM MEAN TO MEDIAN
        results.append({"n": n, "method":"NIV$_{3 \\textrm{ lags}}$", "error": np.mean(error_niv_lags)})
    
    return results


if __name__ == "__main__":
    n_reps = 1000
    results = [i for x in tqdm(Pool(cpu_count()-1).imap_unordered(one_simulation, range(n_reps)), total=n_reps) for i in x]
    df = pd.DataFrame(results)
    df.to_csv("experiments/fig-5-left/results.csv", index=False)

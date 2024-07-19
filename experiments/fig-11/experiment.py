from tqdm import tqdm
from src import simulation, var_iv
from pandas import Series, DataFrame
from src import civ
import pandas as pd
import numpy as np
from multiprocessing import Pool
from itertools import product
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

def ts_iv(X, Y, I, W=None, only_I_as_condition=False):
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(I, (Series, DataFrame)):
        I = I.to_numpy()

    target, lagged = civ.align(
        Y,
        [
            (X, 1),                 # X_{-1} is regressor
            (I, 2),                 # I_{t-2} is instrument
        ],
        tuples=True)

    # Extract relevant sets from lagged time series
    regressor = lagged[0]
    instrument = lagged[1]

    # If no weight matrix is given, compute optimal weight matrix
    if W is None:
        W = var_iv.optimal_weight(
            regressor, target, instrument, conditioning=None)

    return civ.civ(
        X=regressor, Y=target, I=instrument, B=None, W=W
        )



def sample_alphas(skel, instrument_strength=1):
    out = skel * np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)
    for row, col in product(id_X, id_I):
        out[row, col] *= instrument_strength
    return out

def one_simulation(i=None):
    results = []

    for n in map(int, [10**x for x in [2, 2.5, 3, 3.5, 4]]):
        strength = 0.5 / np.sqrt(n)

        # Simulate parameters until stable system
        A1 = sample_alphas(skel, strength)
        while np.max(abs(np.linalg.eigvals(A1)))>1-tol:
            A1 = sample_alphas(skel, strength)
        # Save parameters at lag 1 (gives VAR(1) process)
        A = {1: A1}

        error_civ, error_civ_I3, error_niv_lags, error_niv, error_iv = [], [], [], [], []
        for rep in range(10):
            # Simulate data
            data = simulation.sim_data(A, n=n, Sigma=1.0)
            I, X, Y = data[:,id_I], data[:,id_X], data[:,id_Y]

            beta_0 = A[1][id_Y][:,id_X].T

            # Compute two CIV estimators
            error_civ.append(MSE(var_iv.ts_civ(I=I, X=X, Y=Y) - beta_0))
            error_civ_I3.append(MSE(var_iv.ts_civ(I=I, X=X, Y=Y, only_I_as_condition=True) - beta_0))
            
            # Compute two NIV estimators
            error_niv_lags.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y, n_lags=3) - beta_0))
            error_niv.append(MSE(var_iv.ts_niv(I=I, X=X, Y=Y) - beta_0))
            error_iv.append(MSE(ts_iv(I=I, X=X, Y=Y) - beta_0))


        # Save results
        results.append({"n": n, "method":"CIV$_{I,X,Y}$", "error": np.mean(error_civ), "strength": strength})
        results.append({"n": n, "method":"CIV$_{I}$", "error": np.mean(error_civ_I3), "strength": strength})
        results.append({"n": n, "method":"NIV$_{3 \\textrm{ lags}}$", "error": np.mean(error_niv_lags), "strength": strength})
        results.append({"n": n, "method":"NIV$_{1 \\textrm{ lag}}$", "error": np.mean(error_niv), "strength": strength})
        results.append({"n": n, "method":"IV", "error": np.mean(error_iv), "strength": strength})
    
    return results

results = one_simulation(0)

if __name__ == "__main__":
    n_reps = 1000
    results = [i for x in tqdm(map(one_simulation, range(n_reps)), total=n_reps) for i in x]
    df = pd.DataFrame(results)
    df.to_csv("experiments/fig-11/sqrt-results.csv", index=False)


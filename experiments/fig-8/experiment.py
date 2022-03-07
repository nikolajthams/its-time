import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from src import simulation, var_iv
import gc


# Set dimensions
dI = 1
dX = 1
dH = 1
dY = 1
d = dI + dX + dH + dY

# Get indices for data
id_I, id_X, id_H, id_Y = np.split(np.arange(d),
                                  np.cumsum([dI, dX, dH, dY]))[:4]

# Get skeleton of matrix A
skel = simulation.get_skeleton(id_I, id_X, id_H, id_Y)

# Numerical safety margin for A's eigvals check
tol = 1e-1
sigma = 1.
sigma_pred = sigma

# How many samples
samplesize = 3000
n_matrices = 100
predictions_per_A = 100

# sigmaouts = [1, 3, 5, 7, 9, 11]
sigmaouts = [1, 5]


def rel_error(yhat, y):
    return abs(float(yhat - y)) / abs(float(y))


def sample_alphas():
    # standard parameters
    # A1 = np.random.uniform(-0.9, +0.9, size=skel.shape)

    # bound all away from zero
    A1 = np.random.uniform(0.1, 0.9, size=skel.shape) * \
        np.random.choice([-1, 1], size=skel.shape)

    # bound H-> away from zero
    # alteration here: alpha_i,H bounded away from zero
    A1[:, id_H] = np.random.uniform(0.5, 0.9, size=(skel.shape[0], 1)) \
        * np.random.choice([-1, 1], size=(skel.shape[0], 1))

    # bound I->X away from zero
    # A1[id_X, id_I] = np.random.uniform(0.1, 0.9) * np.random.choice([-1, 1])

    return skel * A1


results = []


def one_simulation(rep=None):
    # Simulate parameters until stable system
    np.random.seed(rep)
    A1 = sample_alphas()
    # clever way to restrict As?
    """
    while (np.max(abs(np.linalg.eigvals(A1))) > 1 - tol or
           np.max(
               np.cov(
                   simulation.sim_data(
                       {1: A1}, n=samplesize, Sigma=sigma).T
               ).diagonal()) > 10):
        A1 = sample_alphas()
    """
    while np.max(abs(np.linalg.eigvals(A1))) > 1 - tol:
        A1 = sample_alphas()

    # Save parameters at lag 1 (gives VAR(1) process)
    A = {1: A1}

    # Obtain interventions relative to observed range
    XTs = [simulation.sim_data(A, n=samplesize, Sigma=sigma)[-1, id_X]
           for _ in range(100)]
    doXs = [#np.random.choice([-1, 1]) * 
            k * np.std(XTs)
            for k in sigmaouts]
    gc.collect()

    # TRAINING
    # Simulate data
    data = simulation.sim_data(A, n=samplesize, Sigma=sigma)
    I, X, Y = data[:, id_I], data[:, id_X], data[:, id_Y]

    # Algorithm 1 for both estimators
    # Causal parameter
    beta_civ = float(var_iv.ts_civ(I=I, X=X, Y=Y))
    beta_niv = float(var_iv.ts_niv(I=I, X=X, Y=Y))
    beta_0 = float(A[1][id_Y][:, id_X].T)
    print(f'sanity check, beta: {beta_0:.4f}, {beta_civ:.4f}, {beta_niv:.4f}')

    # Below we have commented out the variant where we regress with offset term

    # Residual process
    r_civ = Y[3:-2] - beta_civ * X[2:-3]
    r_niv = Y[3:-2] - beta_niv * X[2:-3]
    r_0 = Y[3:-2] - beta_0 * X[2:-3]
    # Regress residuals on past of X and Y
    # (m = 2, l = 1 in Algo 1)
    lagged = np.hstack([
        X[1:-4],
        X[0:-5],
        Y[2:-3],
        Y[1:-4],
    ])
    ols_civ = sm.OLS(r_civ, lagged).fit()
    ols_niv = sm.OLS(r_niv, lagged).fit()
    ols_0 = sm.OLS(r_0, lagged).fit()
    # ols_civ = sm.OLS(r_civ, sm.add_constant(lagged)).fit()
    # ols_niv = sm.OLS(r_niv, sm.add_constant(lagged)).fit()

    # will feed: doX, X[-2], X[-3], Y[-1], Y[-2]

    def civ_pred(covariates):
        return beta_civ*covariates[0] \
            + ols_civ.predict(covariates[1:])
        # + ols_civ.predict(np.r_[[1], covariates[1:]])

    def niv_pred(covariates):
        return beta_niv*covariates[0] \
            + ols_niv.predict(covariates[1:])
        # + ols_niv.predict(np.r_[[1], covariates[1:]])

    def tb_pred(covariates):
        return beta_0*covariates[0] \
            + ols_0.predict(covariates[1:])
        # + ols_0.predict(np.r_[[1], covariates[1:]])

    # OLS
    target = Y[3:-1]
    lagged = np.hstack([
        X[2:-2],
        X[1:-3],
        X[0:-4],
        Y[2:-2],
        Y[1:-3],
    ])
    ols = sm.OLS(target, lagged).fit()
    # ols = sm.OLS(target, sm.add_constant(lagged)).fit()

    def ols_pred(covariates):
        return ols.predict(covariates)
    # return ols.predict(np.r_[[1], covariates])

    # TESTING
    for rep_pred in range(predictions_per_A):
        data = simulation.sim_data(A, n=samplesize, Sigma=sigma)

        # various intervention strengths
        for doX, sigout in zip(doXs, sigmaouts):
            covariates = np.c_[
                doX,
                data[-2, id_X],
                data[-3, id_X],
                data[-1, id_Y],
                data[-2, id_Y],
                ].ravel()

            # obtain predictions
            civ_yhat = civ_pred(covariates)
            niv_yhat = niv_pred(covariates)
            ols_yhat = ols_pred(covariates)
            tb_yhat = tb_pred(covariates)

            # simulate under invervention
            noise = np.random.multivariate_normal(
                mean=np.zeros(data.shape[1]),
                cov=np.eye(data.shape[1]))
            intervened = np.copy(data[-1, :])
            intervened[id_X] = doX
            step = A1 @ intervened + sigma_pred * noise

            ytrue = step[id_Y]

            # log MSE
            results.append({"rep": rep,
                            "sigmaout": sigout,
                            "rep_pred": rep_pred,
                            "method": "CIV",
                            "beta": float((beta_civ - beta_0)**2),
                            "beta_rel": rel_error(beta_civ, beta_0),
                            "error_rel": rel_error(civ_yhat, ytrue),
                            "error": ((civ_yhat - ytrue)**2)[0]})
            results.append({"rep": rep,
                            "sigmaout": sigout,
                            "rep_pred": rep_pred,
                            "method": "NIV",
                            "beta": float((beta_niv - beta_0)**2),
                            "beta_rel": rel_error(beta_niv, beta_0),
                            "error_rel": rel_error(niv_yhat, ytrue),
                            "error": ((niv_yhat - ytrue)**2)[0]})
            results.append({"rep": rep,
                            "sigmaout": sigout,
                            "rep_pred": rep_pred,
                            "method": "TB",
                            "beta": 0,
                            "beta_rel": 0,
                            "error_rel": rel_error(tb_yhat, ytrue),
                            "error": ((tb_yhat - ytrue)**2)[0]})
            results.append({"rep": rep,
                            "sigmaout": sigout,
                            "rep_pred": rep_pred,
                            "method": "OLS",
                            "beta": -999,
                            "beta_rel": -999,
                            "error_rel": rel_error(ols_yhat, ytrue),
                            "error": ((ols_yhat - ytrue)**2)[0]})
    gc.collect()
    return results


if __name__ == "__main__":
    # run k-th bulk of 20 repetitions
    k = 4
    results = [i for x in
               tqdm(
                   Pool(cpu_count()-1).imap_unordered(
                       one_simulation,
                       range(n_matrices)),
                    #    range(0+20*k, 20*(k+1))),
                   total=n_matrices)
                #    total=20)
               for i in x]

    df = pd.DataFrame(results)

    # # take mean over predictions_per_A repetitions per reach A
    # df.groupby(["sigmaout", "method", "rep"]).mean().to_csv(
    #     f"experiment_int_{k}.csv"
    #     # experiments/experiment_int/
    #     )

    # df = pd.concat([
    #     pd.read_csv(
    #         f"experiment_int_{l}.csv")
    #         # experiments/experiment_int/
    #     for l in range(k + 1)])
    df.groupby(["sigmaout", "method", "rep"]).mean().to_csv(
        f"experiments/fig-8/results.csv")

import numpy as np
from pandas import Series, DataFrame
import scipy.linalg as slg
from src import civ
from statsmodels.stats.sandwich_covariance import S_hac_simple


def ts_civ(X, Y, I, W=None, only_I_as_condition=False):
    """
    Compute the ts-civ estimator from observations of time series I, X and Y

    Inputs:
    - X: Regressor time series. nparray shape: (n_obs,) or (n_obs, dims_X)
    - Y: Response time series. nparray shape: (n_obs,) or (n_obs, dims_Y)
    - I: Instrument time series. nparray shape: (n_obs,) or (n_obs, dims_I)
    """
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
            (Y, 1), (X, 2), (I, 3)  # Conditioning sets
        ],
        tuples=True)

    # Extract relevant sets from lagged time series
    regressor = lagged[0]
    instrument = lagged[1]
    conditioning = lagged[4] if only_I_as_condition else civ.col_bind(*lagged[2:])

    # If no weight matrix is given, compute optimal weight matrix
    if W is None:
        W = optimal_weight(
            regressor, target, instrument, conditioning=conditioning)

    return civ.civ(
        X=regressor, Y=target, I=instrument, B=conditioning, W=W)


def ts_niv(X, Y, I, n_lags=None, W=None):
    """
    Compute the ts-niv estimator from observations of time series I, X and Y

    Inputs:
    - X: Regressor time series. nparray shape: (n_obs,) or (n_obs, dims_X)
    - Y: Response time series. nparray shape: (n_obs,) or (n_obs, dims_Y)
    - I: Instrument time series. nparray shape: (n_obs,) or (n_obs, dims_I)
    """
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(I, (Series, DataFrame)):
        I = I.to_numpy()

    # Number of instruments needs to be >= dimension of regressors
    if n_lags is None:
        d_Y, d_X, d_I = (civ.get_2d(v).shape[1] for v in (Y, X, I))
        n_lags = int(np.ceil((d_Y+d_X)/d_I))

    target, lagged = civ.align(
        Y,
        [
            (X, 1),                             # X_{t-1} is regressor
            (Y, 1)                              # Y_{t-1} is nuisance regressor
        ] + [(I, 2+j) for j in range(n_lags)],  # I_{t-2-j} are instruments
        tuples=True)

    # Extract relevant sets from lagged time series
    regressor = lagged[0]
    nuisance = lagged[1]
    instrument = civ.col_bind(*lagged[2:])

    # If no weight matrix is given, compute optimal weight matrix
    if W is None:
        W = optimal_weight(regressor, target, instrument, nuisance)

    return civ.civ(X=regressor, Y=target, I=instrument, N=nuisance, W=W)

def optimal_weight(regressor,
                   target,
                   instrument,
                   nuisance=None,
                   conditioning=None):
    """
    Function to compute optimal weight matrix using the HAC estimator
    """
    # Nuisance is treated as a regressor in the GMM, so add to regressor
    if nuisance is not None:
        regressor = civ.col_bind(regressor, nuisance)

    # Fit initial estimate
    beta_0 = civ.civ(
        X=regressor, Y=target, I=instrument, W=None, B=conditioning)

    # Residual process
    u_t = target - regressor@beta_0

    # Estimand process
    f_t = u_t * instrument

    # Fit VAR(1) process on estimands
    response, covariate = civ.align(f_t, [f_t, 1])

    # Compute A and truncate eigenvalues
    # A = sm.OLS(response, covariate[0]).fit().params
    A = slg.lstsq(covariate[0], response)[0]
    u, s, v = slg.svd(A)
    s = np.clip(s, -0.97, 0.97)
    A = (u*s)@v

    # Residual proces of fitted VAR(1) estimand process
    e_t = response - covariate[0] @ A

    # Get HAC estimator
    S_e = S_hac_simple(e_t)

    # instead of..
    # l = np.linalg.inv(np.eye(S_e.shape[0]) - A)
    # S_hac = l@S_e@l.T
    # return inv(S_hac)
    # ..which does not ensure S_hac is symmetric..

    # ..we now, ensuring symmetry per fiat, do..
    _, s, vh = np.linalg.svd(S_e, hermitian=True)
    eye = np.eye(S_e.shape[0])
    S_hac_inv_sqrt = (eye - A).T @ (vh.T / np.sqrt(s)) @ vh
    S_hac_inv = S_hac_inv_sqrt @ S_hac_inv_sqrt.T
    assert(np.all(S_hac_inv == S_hac_inv.T))

    # ..and also pass along a factorization that may be used downstream..
    return S_hac_inv, S_hac_inv_sqrt

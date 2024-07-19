import numpy as np
import scipy.linalg as slg
import statsmodels.api as sm
import warnings


def posinv(A):
    cholesky, info = slg.lapack.dpotrf(A)
    if info != 0:
        raise np.linalg.LinAlgError('Singular or non-pd Matrix.')
    inv, info = slg.lapack.dpotri(cholesky)
    if info != 0:
        raise np.linalg.LinAlgError('Singular or non-pd Matrix.')
    inv += np.triu(inv, k=1).T
    return inv


# Column and row bind operators
def col_bind(*args):
    return np.concatenate([get_2d(a) for a in args], axis=1)


def row_bind(*args):
    return np.concatenate(args, axis=0)


def get_2d(a):
    """
    Reshape a 1- or 2-d numpy-array to be 2-dimensional
    """
    if len(a.shape) <= 1:
        a = a.reshape(-1, 1)
    return a


def cov(a, b=None):
    """
    Compute cross-covariance matrix between arrays a and b.
    If b is None, covariance matrix of a is returned.

    Inputs:
    - a: Array of shape n_obs OR (n_obs, dims_a)
    - b: None or array of shape n_obs OR (n_obs, dims_b)

    Outputs:
    - Covariance matrix of shape (dims_a, dims_b)
    """
    # Reshape vectors to tall matrices
    a = get_2d(a)
    b = a if b is None else get_2d(b)

    # Extract dimensions
    d_a = a.shape[1]

    # Calculate covariance and return
    Sigma = np.cov(col_bind(a, b).T)
    return Sigma[:d_a, d_a:]


def civ(X: np.ndarray,
        Y: np.ndarray,
        I: np.ndarray,
        B: np.ndarray=None,
        N: np.ndarray=None,
        W: np.ndarray=None,
        return_estimator_covariance: bool=False,
        predict = lambda x,b: sm.OLS(x, b).fit().predict()):
    """
    Compute the causal effect of X on Y using instrument I and conditioning set B.

    Inputs:
    - X:        Regressor. numpy array [shape (n_obs,) OR (n_obs, dims_X)]
    - Y:        Response. numpy array [shape (n_obs,) OR (n_obs, dims_Y)]
    - I:        Instrument. numpy array [shape (n_obs,) OR (n_obs, dims_I)]
    - B:        Conditioning set. numpy array [shape (n_obs,) OR (n_obs, dims_B)]
    - N:        Nuissance regressor. numpy array [shape (n_obs,) OR (n_obs, dims_B)]
    - W:        Weight matrix [shape (dims_I, dims_I)]
                or a tuple (Weight matrix W, Weight matrix factor L s.t. W=LL')
    - predict:  function(X, B) that predicts X from B

    Outputs:
    - Estimated causal effect; numpy array (dims_X, dims_Y)
    """
    # If conditioning set is given, compute prediction residuals of X, Y and I
    if B is not None:
        r_X = get_2d(X) - get_2d(predict(get_2d(X), get_2d(B)))
        r_Y = get_2d(Y) - get_2d(predict(get_2d(Y), get_2d(B)))
        r_I = get_2d(I) - get_2d(predict(get_2d(I), get_2d(B)))
        if N is not None:
            r_N = get_2d(N) - get_2d(predict(get_2d(N), get_2d(B)))
        # Run unconditional IV on residuals
        return civ(
            r_X, r_Y, r_I, B=None, W=W, N=(r_N if N is not None else None),
            return_estimator_covariance=return_estimator_covariance
            )
    # If no conditioning set given, run unconditional IV
    else:
        # Set weight matrix if not given
        if W is None:
            try:
                W = posinv(cov(I))
            except np.linalg.LinAlgError as e:
                e.args += (
                    'Instruments may have degenerate covariance matrix; '
                    'try using less instruments.', )
                warnings.warn(
                    e.args[0]
                    + ' Instruments may have degenerate covariance matrix; '
                    + 'try using less instruments.',
                    slg.LinAlgWarning,
                    stacklevel=3)
                W = np.eye(I.shape[1])
        # Compute weight matrix factor if not already provided
        if type(W) is not tuple:
            W = (W, slg.lapack.dpotrf(W)[0].T)

        regressors = X if N is None else col_bind(X, N)
        covregI = cov(regressors, I)
        covIY = cov(I, Y)
        weights = W[0]
        cho = covregI @ W[1]
        # the following amounts to
        # (covregI @ W @ covregI.T)^(-1) @ covregI @ weights @ covIY
        # while
        # * we ensure symmetry of the covregI @ W @ covregI.T part per fiat
        # * explicitly exploit its positiveness in solve
        # * use solve(A, B) instead of inv(A) @ B for numerical stability
        estimates = slg.solve(cho @ cho.T,
                              covregI @ weights @ covIY,
                              assume_a='pos')


        estimates = estimates if N is None else estimates[:X.shape[1]]
        if return_estimator_covariance:
            covariance = slg.solve(cho @ cho.T,
                            np.eye(cho.shape[0]),
                            assume_a='pos')
            return estimates, covariance[:X.shape[1],:X.shape[1]]
        else:
            return estimates
        


def align(ref, lagged, tuples=False, min_lag = 0):
    """Returns appropriately lagged values for time series data.

    Inputs:
    - ref:          Reference time series, lagged at 0. numpy array of shape (n_obs,) or (n_obs, dims)

    - lagged:       List of time series to be lagged relative to ref.
                    Provide either as [X1, lag1, X2, lag2, ...] or as list of tuples [(X1, lag1), (X2, lag2), ...],
                    where each X is numpy array of shape (n_obs,) or (n_obs, dims) and lag is integer

    - tuples:       Indicate whether lagged is provided as list of tuples or plain list

    Outputs: (ref, lagged)
    - ref:          Reference time series (numpy array, shape (n_obs-m, dims)),
                    with appropriately many entries removed in the beginning to have same length as lagged

    - lagged:       List [X1, X2, ...] of lagged time series, each of shape (n_obs, dims)
    """
    if not tuples:
        it = iter(lagged)
        lagged = list(zip(it, it))
    lagged = [(get_2d(x[:-v, ...]) if v > 0 else x) for (x, v) in lagged]
    m = min(x.shape[0] for x in lagged)
    m = min(m, ref.shape[0] - min_lag)
    lagged = [x[(x.shape[0]-m):, ...] for x in lagged]
    ref = get_2d(ref[(ref.shape[0]-m):, ...])
    return ref, lagged

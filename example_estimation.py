import pandas as pd
from src import var_iv, civ
import numpy as np

# Load data
data = pd.read_csv("example_data.csv")
I = data[["I1", "I2", "I3"]]
X = data[["X1", "X2"]]
Y = data["Y"]

# Use plain vanilla implementations
var_iv.ts_niv(X, Y, I)
var_iv.ts_civ(X, Y, I)

# Or define own NIV estimator..
X, Y, I         = X.to_numpy(), Y.to_numpy(), I.to_numpy()
n_lags          = 4 # Number of instruments used
target, lagged  = civ.align( # Lag regressor, nuisance regressor and instrument relative to Y
    Y,
    [
        (X, 1),                             # X_{t-1} is regressor
        (Y, 1)                              # Y_{t-1} is nuisance regressor
    ] + [(I, 2+j) for j in range(n_lags)],  # I_{t-2-j} are instruments
    tuples=True)
regressor       = lagged[0]
nuisance        = lagged[1]
instrument      = civ.col_bind(*lagged[2:])

# Set weight matrix
W               = np.eye(instrument.shape[1])

# Compute estimator
civ.civ(X=regressor, Y=target, I=instrument, N=nuisance, W=W)
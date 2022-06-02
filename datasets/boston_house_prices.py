import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn import preprocessing


# name
name = 'Boston house prices'
name_saving = 'bostonprices'

# system and simulation parameters
X_tot, y_tot = load_boston(return_X_y=True)
scaler = preprocessing.StandardScaler().fit(X_tot)
X_tot = scaler.transform(X_tot)
nmc = 1000
dimx = X_tot.shape[1]
ndata_max = 450
ndata_min = 50
num_samples = 100
warmup_steps = 100
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)

def get_data(ndata):
    perm = list(np.random.permutation(list(range(506))))
    X_tot, y_tot = load_boston(return_X_y=True)
    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]
    dy = 0.1 * np.random.random(y_data.shape)
    noise = np.random.normal(0, dy)
    y_data += noise

    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]
    return X_data, y_data, X_test, f_true


# ----------------------------------------------------------------------
# GP kernel hyperparameter bounds
ub = 1e2*torch.ones(15)
lb = 1e-1*torch.ones(15)
lb[0] = 1
ub[0] = 1e2
import numpy as np
import torch
"""Module with GP model definitions using various kernels."""

import gpytorch
from gpytorch.priors import UniformPrior
from gpytorch.kernels import (
    SpectralMixtureKernel
)

from sklearn.datasets import fetch_openml

# name
name = 'Mauna loa dataset'
name_saving = 'maunaloa'


# system and simulation parameters
nmc = 300
ndata_min = 50
ndata_max = 300
num_samples = 100
warmup_steps = 50
dimx = 1
datasizes = torch.linspace(ndata_min, ndata_max, 2, dtype=int)

# the first entry corresponds to the mixture weights (signal variances), the second to the
# mixture scales (lengthscales), the last to the noise variance
lb = torch.as_tensor([1e-10, 1e-10, 1e-5])
ub = torch.as_tensor([1e5, 5*1e12, 1e2])

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187, as_frame=False)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs

def get_data(ndata):
    perm = torch.as_tensor(list(np.random.permutation(list(range(500)))))
    X_tot, y_tot = load_mauna_loa_atmospheric_co2()
    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]

    dy = 0.5*ub[-1] * np.random.random(y_data.shape)
    noise = np.random.normal(0, dy)
    y_data += noise

    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]

    return X_data, y_data, X_test, f_true


kernel = SpectralMixtureKernel(10)
kernel.register_prior('mixture_scales_prior', UniformPrior(lb[1], ub[1]), 'mixture_scales')
kernel.register_prior('mixture_weights_prior', UniformPrior(lb[0], ub[0]), 'mixture_weights')

kernel.register_constraint('raw_mixture_scales',gpytorch.constraints.Interval(lb[1], ub[1]))
kernel.register_constraint('raw_mixture_weights',gpytorch.constraints.Interval(lb[0], ub[0]))


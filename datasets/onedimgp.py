import torch
import numpy as np
import  gpytorch
from gpregression.ExactGPModel import ExactGPModel

# name
name = 'One-dimensional Gaussian process:'
name_saving = 'gaussianprocess'


# system and simulation parameters
dimx = 1
ntest = 301
ndata_max = 6
ndata_min = 2
num_samples = 150
warmup_steps = 150
nreps = 1
torquenr = 0 # specifies which torque is to be inferred (0-6)
datasizes = torch.linspace(ndata_min,ndata_max,3,dtype=int)

# Training data is 100 points in [0,1] inclusive regularly spaced
x_tot = torch.linspace(0, 1, 301)


# ----------------------------------------------------------------------
# set upper and lower bounds for uniform hyperprior
ub = 7*torch.ones(dimx+2)
ub[0] = 7
ub[-1] = 0.13
lb = 1e-2*torch.ones(dimx+2)
lb[-1] = 1e-4



perm = list(np.random.permutation(list(range(x_tot.shape[0]))))
perm[0:9] = 4*[3, 19,  225, 269, 101, 140, 350, 352, 355, 356]

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise_covar._set_noise(0.0043)
# use single point to initialize GP
y_tr = torch.tensor([1.1109])
model = ExactGPModel(x_tot[50:51], y_tr, likelihood, dimx, lb=lb, ub=ub)
model.covar_module.base_kernel.lengthscale = 0.284
model.covar_module.outputscale = 2.5

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()


# Create test set and condition GP model on it
y_tot = model(x_tot).sample()
X_test = x_tot
f_true = y_tot

def get_data(ndata):
    # Load training set
    X_data = x_tot[perm[:ndata]]
    y_data = y_tot[perm[:ndata]]

    return X_data, y_data, X_test, f_true

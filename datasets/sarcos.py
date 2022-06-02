import scipy.io
import torch
import numpy as np

# name
name = 'Sarcos dataset'
name_saving = 'sarcos'

# system and simulation parameters
dimx = 21
nreps = 100
ntest = 4000
ndata_max = 10000
ndata_min = 800
num_samples = 100
warmup_steps = 100
training_iterations = 10000 # 10000
ndatanumbers = 3
torquenr = 0  # specifies which torque is to be inferred (0-6)

datasizes = [800, 5000, 10000]

def get_data(ndata):
    # Load training set
    train = scipy.io.loadmat("datasets/sarcos_inv.mat")
    perm = torch.as_tensor(list(np.random.permutation(list(range(44484)))))
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    X_data = train["sarcos_inv"][perm[:ndata], :21]
    # Outputs (7 joint torques)
    y_data = train["sarcos_inv"][perm[:ndata], 22 + torquenr]

    # Load test set
    test = scipy.io.loadmat("datasets/sarcos_inv_test.mat")
    X_test = test["sarcos_inv_test"][:ntest, :21]
    f_true = test["sarcos_inv_test"][:ntest, 22 + torquenr]

    return X_data, y_data, X_test, f_true


# ----------------------------------------------------------------------

# set upper and lower bounds for unifor hyperprior
ub = 1000 * torch.ones(dimx + 2)
ub[1:-1] = 50
ub[-1] = 80
lb = 1e-1 * torch.ones(dimx + 2)
lb[0] = 1e-1


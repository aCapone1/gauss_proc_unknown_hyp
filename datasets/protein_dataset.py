import os
import torch
import numpy as np
import sys
# sys.path.append('../')
import urllib.request
from scipy.io import loadmat
dataset = 'protein'
if not os.path.isfile(f'./datasets/{dataset}.mat'):
    print(f'Downloading \'{dataset}\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1nRb8e7qooozXkNghC5eQS0JeywSXGX2S',
                               f'./datasets/{dataset}.mat')

data = torch.Tensor(loadmat(f'./datasets/{dataset}.mat')['data'])

# name
name = 'Protein dataset'
name_saving = 'protein'

# system and simulation parameters
dimx = 9
nreps = 10
ntest = 4000
ndata_max = 30000
ndata_min = 9000
num_samples = 100
warmup_steps = 100
training_iterations = 10000 # 10000

N = data.shape[0]
# make train/val/test
n_train = int(0.8 * N)
x_train, y_train = data[:n_train, :-1], data[:n_train, -1]
x_test, y_test = data[n_train:, :-1], data[n_train:, -1]

# normalize features
meanx = x_train.mean(dim=-2, keepdim=True)
stdx = x_train.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
x_train = (x_train - meanx) / stdx
x_test = (x_test - meanx) / stdx

# normalize labels
meany, stdy = y_train.mean(),y_train.std()
y_train = (y_train - meany) / stdy
y_test = (y_test - meany) / stdy

def get_data(ndata):

    perm = torch.as_tensor(list(np.random.permutation(list(range(36584)))),dtype=torch.long)
    # Load training set
    X_data = x_train[perm[:ndata]]
    y_data = y_train[perm[:ndata]]

    # Load test set
    X_test = x_test
    f_true = y_test

    return X_data, y_data, X_test, f_true


# ----------------------------------------------------------------------

# set upper and lower bounds for unifor hyperprior
ub = 1000 * torch.ones(dimx + 2)
ub[1:-1] = 150
ub[-1] = 200
lb = 10 * torch.ones(dimx + 2)
lb[0] = 1e-1

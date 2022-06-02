import gpytorch
import torch
import gc
import numpy as np
from copy import deepcopy
from functions.LBFGS import FullBatchLBFGS


def train(train_x, train_y, model0, likelihood0, n_training_iter):
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)
    for i in range(n_training_iter):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output0 = model0(train_x)
        # Calc loss and backprop derivatives
        loss = -mll0(output0, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f \r' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()
    return model0, likelihood0, mll0(output0, train_y)


def traingpu(train_x, train_y, model0, likelihood0, checkpoint_size,
             preconditioner_size, n_training_iter):
    model0.train()
    likelihood0.train()

    optimizer = FullBatchLBFGS(model0.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)

    # setting gpytorch.settings.fast_computations(log_prob=False) will cause the log likelihood to be deterministic,
    # which will speed up training at the expense of a low-quality solution
    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(preconditioner_size), \
            gpytorch.settings.max_cg_iterations(100000): #, \
            # gpytorch.settings.fast_computations(log_prob=False):

        def closure():
            optimizer.zero_grad()
            output = model0(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 100}
            loss, grad, _, _, _, _, _, fail = optimizer.step(options)

            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_training_iter, loss.item()), end='\r')
            # print('lengthscales:')
            # print([ls.item() for ls in model0.covar_module.module.base_kernel.lengthscale[0]])
            # print('noise: %.3f' % model0.likelihood.noise.item())

            if fail:
                print('\nConvergence reached!')
                break

        def f(x):
            model = deepcopy(model0)
            # monkey patch substitute variables
            named_params = list(model.named_parameters())
            for name in named_params:
                print(name)

    return model0, likelihood0, -loss.detach().item()


def find_best_gpu_setting(train_x, train_y, model0, likelihood0, n_devices, output_device, preconditioner_size):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _, _ = traingpu(train_x, train_y, model0=model0, likelihood0=likelihood0,
                               checkpoint_size=checkpoint_size,
                               preconditioner_size=preconditioner_size, n_training_iter=1)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size

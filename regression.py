"This code is partially based on the Exact GP Regression with Multiple GPUs and \
Kernel Partitioning Example, which can be found at \
https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html"

import pickle
import torch
import random
import gpytorch
import pyro
import numpy as np
from functions.plot1D import plot1D

from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.priors import UniformPrior

from functions import *
from functions.plot1D import plot1D
from gpregression.ExactGPModel import ExactGPModel

np.random.seed(2)
random.seed(2)
torch.manual_seed(0)

nreps = 10 # change to higher number to perform more repetitions
sqrbeta0 = 1.414
training_iterations = 2000  # number of gp training interations
nlaplace = 801  # number of data points after which Laplace approximation is used
kernel = False

print ("Which experiment would you like to run? \n [T]oy problem, [B]oston house prices, [M]auna Loa, [W]ine or [S]arcos?")
while True:
    user_input = input()
    if user_input == "B":
        print ("You have chosen to run the Boston house prices experiment.")
        from datasets.boston_house_prices import *
        break
    elif user_input == "M":
        print ("You have chosen to run the Mauna Loa CO2 experiment.")
        from datasets.mauna_loa import *
        break
    elif user_input == "W":
        print ("You have chosen to run the wine quality experiment.")
        from datasets.wine import *
        break
    elif user_input == "S":
        print ("You have chosen to run the Sarcos experiment.")
        from datasets.sarcos import *
        break
    elif user_input == "T":
        print ("You have chosen to run the toy experiment.")
        from datasets.onedimgp import *
        break
    else:
        print("Please enter T, B, M, W or S and press Enter.")
        continue


# some initializations
perc_base = [None] * nreps
err_offset_base = [None] * nreps
err_base = [None] * nreps
perc_robust = [None] * nreps
err_offset_robust = [None] * nreps
err_robust = [None] * nreps
perc_fullb = [None] * nreps
err_offset_fullb = [None] * nreps
err_fullb = [None] * nreps
gamma = [None] * nreps
loglikelihood0 = [None] * nreps
r0 = []


ndatait = 0

for ndata in datasizes:

    print('Name: ' + name + '. Data set size N=%d' % ndata)
    if ndata > nlaplace:
        # check if GPU is available for large datasets and Laplace approximation
        gpu = torch.cuda.is_available()
        n_devices = 1 #torch.cuda.device_count()
        print('Planning to run on {} GPUs.'.format(n_devices))
        if gpu:
            output_device = torch.device('cuda:0')
        else:
            output_device = torch.device('cpu')
        # transfer bounding tensors of uniform distribution to output device (gpu)
        lb = lb.to(output_device)
        ub = ub.to(output_device)
    else:
        gpu = False

    for rep in list(range(nreps)):

        X_data, y_data, X_test, f_true = get_data(ndata)

        # set up likelihood
        likelihood0 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(lb[-1], ub[-1]),
            noise_prior=UniformPrior(lb[-1], ub[-1]))

        if gpu:
            base_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                           lengthscale_prior=UniformPrior(lb[1:-1],
                                                                          ub[1:-1]),
                                           lengthscale_constraint=gpytorch.constraints.Interval(
                                               lb[1:-1], ub[1:-1])),
                outputscale_prior=UniformPrior(lb[0], ub[0]),
                outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0]))

            kernel = gpytorch.kernels.MultiDeviceKernel(base_kernel, device_ids=range(n_devices),
                                                        output_device=output_device)

            # make continguous
            train_x = torch.Tensor(X_data).detach().contiguous().to(output_device)
            train_y = torch.Tensor(y_data).detach().contiguous().to(output_device)
            test_x = torch.Tensor(X_test).detach().contiguous().to(output_device)
            test_y = torch.Tensor(f_true).detach().contiguous().to(output_device)

            # generate and train vanilla gp with log-likelihood optimization
            model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, lb, ub, kernel).to(output_device)
        else:
            train_x = torch.Tensor(X_data).detach()
            train_y = torch.Tensor(y_data).detach()
            test_x = torch.Tensor(X_test).detach()
            test_y = torch.Tensor(f_true).detach()
            # generate and train vanilla gp with log-likelihood optimization
            model0 = ExactGPModel(train_x, train_y, likelihood0, dimx, lb, ub, kernel)

        print('Training GP model with (constrained) log-likelihood optimization...')

        if gpu:
            # use multiple GPUs if specified
            from gpregression.train import traingpu, find_best_gpu_setting

            likelihood0 = likelihood0.to(output_device)
            model0 = model0.to(output_device)
            preconditioner_size = 100
            checkpoint_size = find_best_gpu_setting(train_x, train_y, model0=model0, likelihood0=likelihood0,
                                                    n_devices=n_devices, output_device=output_device,
                                                    preconditioner_size=preconditioner_size)
            model0, likelihood0, loglikelihood0[rep] = \
                traingpu(train_x, train_y, model0, likelihood0, checkpoint_size=checkpoint_size,
                         preconditioner_size=100, n_training_iter=training_iterations)
        else:
            # training without gpu
            from gpregression.train import train

            model0, likelihood0, loglikelihood0[rep] = \
                train(train_x, train_y, model0, likelihood0, training_iterations)

            checkpoint_size = 0
            preconditioner_size = 100

        print('Name: ' + name_saving + '. Data set size N=%d, Repetition number %d' % (ndata, rep))

        if gpu:
            print('Vanilla GP variance: ' + '{:.4}'.format(model0.covar_module.base_kernel.outputscale.item()))
            print('Vanilla GP lengthscales:')
            print(['{:.4}'.format(ls.detach().item()) for ls
                   in model0.covar_module.base_kernel.base_kernel.lengthscale.data[0]])
        elif model0.covar_module._get_name() == 'SpectralMixtureKernel':
            print('Vanilla GP mixture weights (variances):')
            print(['{:.4}'.format(ws.detach().item()) for ws in model0.covar_module.mixture_weights])
            print('Vanilla GP mixture scales (lengthscales):')
            print(['{:.4}'.format(sc.detach().item()) for sc in model0.covar_module.mixture_scales])
        else:
            print('Vanilla GP variance: ' + '{:.4}'.format(model0.covar_module.outputscale.data))
            print('Vanilla GP lengthscales:')
            print(['{:.4}'.format(ls.detach().item()) for ls  in model0.covar_module.base_kernel.lengthscale.data[0]])

        print('Vanilla GP noise variance: ' + '{:.4}'.format(model0.likelihood.noise_covar.noise.item()))

        # check if data set is too large for fully Bayesian approach. If so, Laplace approximation will be used
        if ndata < nlaplace:

            # set up likelihood
            likelihood = deepcopy(likelihood0)

            # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
            fullbmodel = ExactGPModel(train_x, train_y, likelihood, dimx, lb, ub, deepcopy(kernel))

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, fullbmodel)


            def pyro_model(x, y):
                with gpytorch.settings.fast_computations(False, False, False):
                    sampled_fullbmodel = fullbmodel.pyro_sample_from_prior()
                    output = sampled_fullbmodel.likelihood(sampled_fullbmodel(x))
                    pyro.sample("obs", output, obs=y)
                return y


            nuts_kernel = NUTS(pyro_model)
            mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)
            print('Generating GP samples for fully Bayesian GP...')
            mcmc_run.run(train_x, train_y)

            fullbmodel.pyro_load_from_samples(mcmc_run.get_samples())

            # get marginal log likelihood of models used for MC integration
            if dimx > 1:
                expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
                expanded_test_y = test_y.unsqueeze(0).repeat(num_samples, 1)
            else:
                expanded_test_x = deepcopy(test_x)
                expanded_test_y = deepcopy(test_y)
            fullbmodel.eval()
            outputfb = fullbmodel(expanded_test_x)

            # get bounding gp
            if name_saving == 'maunaloa' or name_saving == 'maunaloatight':
                boundinggp, sqrbetabar, gammaopt = getboundinggp_sm(fullbmodel, model0, [], [])
            else:
                boundinggp, sqrbetabar, gammaopt = getboundinggp(fullbmodel, model0, [], [])
            # evaluation model for GP model
        else:
            print('Using Laplace approximation to compute bounding hyperparameters...')

            # evaluation model for GP model
            mll0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)
            boundinggp, sqrbetabar, gammaopt = getboundinggp_lap(model0, likelihood0, train_x, train_y, [], [],
                                                                 preconditioner_size, checkpoint_size)
            fullbmodel = []
            del mll0
            torch.cuda.empty_cache()

        if gpu:
            print('Bounding GP variance: ' + '{:.4}'.format(boundinggp.covar_module.base_kernel.outputscale.item()))
            print('Bounding GP lengthscales:')
            print(['{:.4}'.format(ls.detach().item()) for ls
                   in boundinggp.covar_module.base_kernel.base_kernel.lengthscale.data])
        elif boundinggp.covar_module._get_name() == 'SpectralMixtureKernel':
            print('Bounding GP mixture weights (variances):')
            print(['{:.4}'.format(ws.detach().item()) for ws in boundinggp.covar_module.mixture_weights])
            print('Bounding GP mixture scales (lengthscales):')
            print(['{:.4}'.format(sc.detach().item()) for sc in boundinggp.covar_module.mixture_scales])
        else:
            print('Bounding GP variance: ' + '{:.4}'.format(boundinggp.covar_module.outputscale.data))
            print('Bounding GP lengthscales:')
            print(['{:.4}'.format(ls.detach().item()) for ls in boundinggp.covar_module.base_kernel.lengthscale.data[0]])

        print('Bounding GP noise variance: ' + '{:.4}'.format(boundinggp.likelihood.noise_covar.noise.item()))

        # generate predictions on test_x. Here we ignore sqrbetabar and employ sqrbeta0 instead
        model0.eval()
        boundinggp.eval()

        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
                gpytorch.settings.max_preconditioner_size(preconditioner_size), \
                gpytorch.settings.max_cg_iterations(1000000), \
                gpytorch.settings.fast_computations(log_prob=False):
            try:
                mean0 = model0(test_x).mean.detach()
                stddev0 = model0(test_x).stddev.detach()
            except:
                # compute each entry of mean and covariance individually if memory limitations are reached
                mean0, stddev0 = torch.as_tensor([model0(x.reshape(1, dimx)).mean.detach()
                                         for x in test_x]).detach().to(output_device), \
                                 torch.as_tensor([model0(x.reshape(1, dimx)).stddev.detach()
                                           for x in test_x]).detach().to(output_device)
            perc_base[rep], err_offset_base[rep], err_base[rep] = evalperf(test_y, mean0, stddev0, mean0, sqrbeta0)

            del model0
            torch.cuda.empty_cache()

            boundinggp.eval()

            try:
                mean_bound, stddev_bound = boundinggp(test_x).mean.detach(), boundinggp(test_x).stddev.detach()
            except:
                mean_bound, stddev_bound = torch.as_tensor([boundinggp(x.reshape(1, dimx)).mean.detach()
                                                for x in test_x]).detach().to(output_device), \
                                            torch.as_tensor([boundinggp(x.reshape(1, dimx)).stddev.detach()
                                                             for x in test_x]).detach().to(output_device)
            perc_robust[rep], err_offset_robust[rep], err_robust[rep] = evalperf(test_y, mean0, stddev_bound,
                                                                                 mean_bound, sqrbeta0)

            del boundinggp
            torch.cuda.empty_cache()

        if fullbmodel:
            # transpose means and standard deviations of fully Bayesian model for Gaussian mixture model
            fbmeans = torch.transpose(outputfb.mean,0,1)
            fbstddevs = torch.transpose(outputfb.stddev,0,1)

            # create set of normally distributed variables with means fbmeans and standard deviations fbstddevs
            fbN = torch.distributions.Normal(fbmeans,fbstddevs)

            # generate weights (ones) for Gaussian mixture models
            fbgmmweights = torch.distributions.Categorical(torch.ones(num_samples,))

            # create Gaussian mixture model using fully Bayesian GP evaluations
            fbgmm = torch.distributions.mixture_same_family.MixtureSameFamily(fbgmmweights, fbN)

            perc_fullb[rep], err_offset_fullb[rep], err_fullb[rep] = \
                evalperf(test_y, fbgmm.mean, fbgmm.stddev,
                         fbgmm.mean, sqrbeta0)

            del fullbmodel
            torch.cuda.empty_cache()
            mean_perc_fullb = str('{:.4}'.format(torch.mean(torch.as_tensor(perc_fullb[:rep + 1])).item()))
            mse_fullb = str('{:.4}'.format(torch.mean(torch.as_tensor(torch.stack(err_fullb[:rep + 1]))).item()))
        else:
            mean_perc_fullb = '-'
            mse_fullb = '-'
        gamma[rep] = gammaopt

        datastr = np.array2string(ndata.detach().numpy())


        print('ECE robust GP:  %.3f, ECE vanilla GP: %.3f, ECE fully Bayes GP: ' %
              (torch.mean(torch.as_tensor(perc_robust[:rep + 1])),
               torch.mean(torch.as_tensor(perc_base[:rep + 1])))
              + mean_perc_fullb)

        # uncomment the following lines to see MSE of the different models
        # print('MSE robust GP:  %.3f, MSE vanilla GP: %.3f, MSE fully Bayes GP: ' %
        #       (torch.mean(torch.as_tensor(torch.stack(err_robust[:rep + 1]))),
        #        torch.mean(torch.as_tensor(torch.stack(err_base[:rep + 1]))))
        #       + mse_fullb)

        with open('regressionresults/percanderrs' + name_saving + datastr + '.pkl', 'wb') as f:
            pickle.dump([perc_base[:rep], err_offset_base[:rep], err_base[:rep],
                         perc_robust[:rep], err_offset_robust[:rep], err_robust[:rep],
                         perc_fullb[:rep], err_offset_fullb[:rep], err_fullb[:rep], gamma[:rep]], f)

        if name_saving == 'gaussianprocess':

            plot1D(X_data, y_data, X_test, mean0, sqrbeta0, stddev0, sqrbeta0,
                   stddev_bound, fbgmm.mean.detach(), fbgmm.stddev.detach(),
                   f_true)

    ndatait += 1

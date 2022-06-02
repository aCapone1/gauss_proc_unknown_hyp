from scipy.integrate import odeint
from backstfuns import manip
from backstfuns.plot import plot_backstepping_results
import numpy as np
import torch
import pyro
import gpytorch
from pyro.infer.mcmc import NUTS, MCMC
from copy import deepcopy
import pickle

np.random.seed(2)


# observations
from functions.getboundinggp import getboundinggp

nreps = 500 # 50 instead
num_samples = 50
warmup_steps = 50
sols_class = [None]*nreps
sols_safe = [None]*nreps
sols_fb = [None]*nreps

errs_class = []
errs_safe = []
errs_fb = []

for rep in list(range(nreps)):
    X_data, Y_data = manip.get_data()

    dimx = X_data.shape[1]
    C = np.ones(dimx)

    # GP kernel
    nmc = 100
    delta_max = 0.1
    gps = []
    gps_safe = []
    gps_fb = []
    r0 = 0.1

    # We will use the simplest form of GP model, exact inference
    from gpregression.ExactGPModel import ExactGPModel

    from gpytorch.priors import UniformPrior

    for dim in list(range(dimx)):

        # set upper and lower bounds for uniform hyperprior
        ub = 1e-2 * torch.ones(dim + 1 + 2)
        ub[0] = 1e1
        ub[-1] = 1e-1
        lb = 1e-15 * torch.ones(dim + 1 + 2)
        lb[0] = 1e-6
        lb[-1] = 1e-5

        train_x = torch.Tensor(X_data[:,0:dim+1]).detach()
        train_y = torch.Tensor(Y_data[:,dim]).detach()

        # set up likelihood
        likelihood0 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(lb[-1], ub[-1]))
        likelihood0.register_prior("noise_prior", UniformPrior(lb[-1], ub[-1]), "noise")

        # generate and train vanilla gp with log-likelihood optimization
        model0 = ExactGPModel(train_x, train_y, likelihood0, dim+1, lb, ub)

        # Use the adam optimizer
        optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)

        print('Training GP model with (constrained) log-likelihood optimization...')

        training_iterations = 800
        def train():
            for i in range(training_iterations):
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


        train()

        # set up likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        likelihood.register_prior("noise_prior", UniformPrior(lb[-1], ub[-1]), "noise")

        # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
        fullbmodel = ExactGPModel(train_x, train_y, likelihood, dim+1, lb, ub)

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
        fullbmodel.eval()
        model0.eval()

        boundinggp, sqrbetabar, gammaopt = getboundinggp(fullbmodel, model0, [], [])

        print("Vanilla GP variance: " + '{:.3}'.format((model0.covar_module.outputscale.detach().item())).__str__())
        print("Vanilla GP lengthscales: ")
        print(['{:.3}'.format(ls.detach().item()) for ls in model0.covar_module.base_kernel.lengthscale.data[0]])
        print("Vanilla GP noise variance: " + '{:.3}'.format(model0.likelihood.noise.detach().item()).__str__())

        print("Bounding GP variance: " + '{:.3}'.format((boundinggp.covar_module.outputscale.detach().item())).__str__())
        print("Bounding GP lengthscales: ")
        print(['{:.3}'.format(ls.detach().item()) for ls in boundinggp.covar_module.base_kernel.lengthscale.data[0]])
        print("Bounding GP noise variance: " + '{:.3}'.format(boundinggp.likelihood.noise.detach().item()).__str__())

        gps.append(deepcopy(model0))
        gps_safe.append(deepcopy(boundinggp))
        gps_fb.append(deepcopy(fullbmodel))


    x0 = torch.zeros(9)
    x0[0] = 10*np.random.normal()
    x0[1] = 10*np.random.normal() #2 * np.pi +
    x0[2] =  10*np.random.normal()
    err_des = 1
    beta = 2 #np.sqrt(2)
    C_class = 1
    C_safe = 10

    t = np.linspace(0, 100.5, 201) # 50.5 instead of 1.5

    print('Carrying out simulation with robust GP bound...')
    sols_safe[rep] = odeint(manip.manip, x0, t, args=(C*C_safe, gps, gps_safe, beta, err_des)) #C*C_safe

    print('Carrying out simulation with fully Bayesian GP...')
    sols_fb[rep] = odeint(manip.manip, x0, t, args=(C * C_class, gps_fb, gps_fb, beta, err_des))

    print('Carrying out simulation with vanilla GP bound...')
    sols_class[rep] = odeint(manip.manip, x0, t, args=(C*C_class, gps, gps, beta, err_des)) #C*C_class

    xdes_class = []
    xdes_safe = []
    xdes_fb = []
    for step in list(range(t.shape[0])):
        xdes_class.append(manip.get_xd(sols_class[rep][step], t[step]))
        xdes_safe.append(manip.get_xd(sols_safe[rep][step], t[step]))
        xdes_fb.append(manip.get_xd(sols_safe[rep][step], t[step]))

    errs_class.append(sols_class[rep][:,0:3] - xdes_class)
    errs_safe.append(sols_safe[rep][:,0:3] - xdes_safe)
    errs_fb.append(sols_fb[rep][:, 0:3] - xdes_fb)

    with open('backstres/results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([errs_class, errs_safe, errs_fb, sols_class, sols_safe, sols_fb, t], f) #gps, gps_safe, gps_fb,

    try:
        plot_backstepping_results(rep)
    except:
        print('Simulation yielded numerically unstable values. This message should disappear after a few repetitions.')
#
# err_norms_class = [np.sqrt(np.sum(error[:,0:3]**2,1)) for error in errs_class[0:-1]]
# err_norms_safe = [np.sqrt(np.sum(error[:,0:3]**2,1)) for error in errs_safe[0:-1]]
#
# median_err_class = np.median(err_norms_class,0)
# lwdecile_class = np.quantile(err_norms_class,0.1,axis=0)
# updecile_class = np.quantile(err_norms_class,0.9,axis=0)
#
# median_err_safe = np.median(err_norms_safe,0)
# lwdecile_safe = np.quantile(err_norms_safe,0.1,axis=0)
# updecile_safe = np.quantile(err_norms_safe,0.9,axis=0)
#
# with open('backstres/results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([errs_class, errs_safe, err_norms_class, err_norms_safe, median_err_class, lwdecile_class, updecile_class,
#                  median_err_safe, lwdecile_safe, updecile_safe, t], f)

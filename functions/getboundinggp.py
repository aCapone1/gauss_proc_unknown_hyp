from copy import deepcopy
import torch
import gpytorch


def getboundinggp_lap(model0, likelihood0, train_x, train_y, delta_max, sqrbeta, preconditioner_size, checkpoint_size):
    # Get bounding GP parameters using laplace approximation for marginal log likelihood.
    # Only works for squared-exponential kernel.

    # set confidence parameter if not available
    if not delta_max:
        delta_max = 0.1
    if not sqrbeta:
        sqrbeta = 1.414

    # initialize noise covariance and kernel
    noise_covar = deepcopy(model0.likelihood.noise_covar)
    if model0.covar_module._get_name() == 'MultiDeviceKernel':
        kernel = deepcopy(model0.covar_module.base_kernel)
        dev = torch.device('cuda:0')
    elif torch.cuda.is_available():
        dev = torch.device('cuda:0')
        kernel = deepcopy(model0.covar_module)
    else:
        dev = torch.device('cpu')
        kernel = deepcopy(model0.covar_module)

    # extract raw parameters at log likelihood maximum
    raw_outputscale0 = kernel.raw_outputscale  # corresponds to the signal variance (sigma_f^2)
    raw_lengthscale0 = kernel.base_kernel.raw_lengthscale  # lengthscale (NOT logarithm or square)
    raw_noise0 = noise_covar.raw_noise  # noise variance (\sigma_n^2)

    dimx = raw_lengthscale0.shape[1]
    raw_theta0 = [raw_outputscale0.reshape(1, 1), raw_lengthscale0.reshape(dimx, 1), raw_noise0.reshape(1, 1)]
    raw_theta0 = torch.cat(raw_theta0, 0).reshape(dimx + 2, )

    # create copies of model and likelihood that will be used to determine bounding hyperparameters
    model = deepcopy(model0)
    likelihood = deepcopy(likelihood0)
    model.train()
    likelihood.train()

    # deactivate stochastic computations using log_prob=False to be able to compute Hessian
    with gpytorch.settings.deterministic_probes(True), \
            gpytorch.settings.fast_computations(log_prob=False), \
            gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(preconditioner_size), \
            gpytorch.settings.max_cg_iterations(1000000):

        # log likelihood function. Used to compute Hessian
        def loglik(hyperparams):

            modelnew = deepcopy(model0)
            liknew = deepcopy(likelihood0)

            # check if base kernel is wrapped in another kernel and set hyperparameters
            # deleting model parameters using del is necessary to compute hessian with autograd
            if modelnew.covar_module.base_kernel.has_lengthscale:
                del modelnew.covar_module.raw_outputscale  # torch.nn.Parameter(x[0])
                del modelnew.covar_module.base_kernel.raw_lengthscale
                modelnew.covar_module.raw_outputscale = hyperparams[0]  # torch.nn.Parameter(x[0])
                modelnew.covar_module.base_kernel.raw_lengthscale = hyperparams[1:-1]
            else:
                del modelnew.covar_module.base_kernel.raw_outputscale
                del modelnew.covar_module.base_kernel.base_kernel.raw_lengthscale
                modelnew.covar_module.base_kernel.raw_outputscale = hyperparams[0]  # torch.nn.Parameter(x[0])
                modelnew.covar_module.base_kernel.base_kernel.raw_lengthscale = hyperparams[1:-1]  # torch.nn.Parameter(x[1:-1])

            del modelnew.likelihood.noise_covar.raw_noise
            del liknew.noise_covar.raw_noise
            modelnew.likelihood.noise_covar.raw_noise = hyperparams[-1].reshape(1,).to(dev)  # torch.nn.Parameter(x[-1].reshape(1,))
            liknew.noise_covar.raw_noise = hyperparams[-1].reshape(1,).to(dev)

            mllnew = gpytorch.mlls.ExactMarginalLogLikelihood(liknew, modelnew).to(dev)
            mllnew.zero_grad()
            logl = mllnew(modelnew(train_x), train_y)

            return logl

        # extract number of hyperparameters
        npar = raw_theta0.shape[0]
        # compute hessian using autograd package
        hessloglik = torch.autograd.functional.hessian(loglik, torch.autograd.Variable(raw_theta0).to(dev))
        # regularize using curvature to guarantee that Hessian is negative definite.
        # Invert and negate to obtain covariance matrix
        curvature = torch.max(torch.tensor(1e-2),
                              10*max(torch.real(torch.linalg.eigvals(hessloglik)))+torch.tensor(1e-2))
        # uncomment to show curvature
        # print('Curvature of quadratic hyperprior:')
        # print(curvature.item())
        invnegh = -torch.inverse(hessloglik - curvature*torch.eye(npar).to(dev))
        # print('Diagonal entries of inverse negative Hessian:')
        # print(torch.diagonal(invnegh).detach())

    # introduce hard constraint on conf_new to avoid excessively conservative estimates
    conf_new = torch.min(torch.tensor(0.9), torch.pow(1 - delta_max, torch.tensor([1 / npar]))).to(dev)

    # initialize raw bounding hyperparameters
    raw_thprim = torch.zeros(raw_theta0.shape).to(dev)
    raw_thdoubprim = torch.zeros(raw_theta0.shape).to(dev)

    i = 0
    for sigsq in torch.diagonal(invnegh):
        logmult = torch.distributions.Normal(0, 1).icdf(torch.tensor([conf_new]).to(dev)).to(dev)
        delta = torch.exp(logmult).to(dev) * torch.sqrt(sigsq.clone().detach()).to(dev)
        raw_thprim[i] = raw_theta0[i] - delta
        raw_thdoubprim[i] = raw_theta0[i] + delta
        i += 1

    torch.cuda.empty_cache()

    # Create robust bounding hyperparameters. These correspond to the minimal lengthscales
    # and maximal signal/noise variances. Referred to as theta' in the experimental sectoin of the paper
    raw_throbust = raw_thprim  # deepcopy(thprimnew)
    raw_throbust[0] = raw_thdoubprim[0]  # deepcopy(thdoubprimnew[0])
    raw_throbust[-1] = raw_thdoubprim[-1]  # deepcopy(thdoubprimnew[-1])

    robustmodel = deepcopy(model0)
    if model0.covar_module._get_name() == 'MultiDeviceKernel':
        robustmodel.covar_module.base_kernel.base_kernel.raw_lengthscale = \
            torch.nn.Parameter(raw_throbust[1:-1])
        robustmodel.covar_module.base_kernel.raw_outputscale = \
            torch.nn.Parameter(raw_throbust[0])
        robustmodel.likelihood.noise_covar.raw_noise = torch.nn.Parameter(raw_throbust[-1].reshape(1,))
    else:
        robustmodel.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(raw_throbust[1:-1])
        robustmodel.covar_module.raw_outputscale = torch.nn.Parameter(raw_throbust[0])
        robustmodel.likelihood.noise_covar.raw_noise = torch.nn.Parameter(raw_throbust[-1].reshape(1,))

    # set gamma to 1
    gamma = 1

    return robustmodel, sqrbeta, gamma


def getboundinggp(sampmods, model0, nmc, delta_max):

    # set number of MCMC samples and delta if not available
    if not nmc:
        nmc = sampmods.covar_module.base_kernel.lengthscale.shape[0]
    if not delta_max:
        delta_max = 0.05

    # extract input dimension from lengthscales
    dimx = sampmods.covar_module.base_kernel.lengthscale.shape[-1]

    # concatenate hyperparameter samples generated with NUTS
    outputscale = sampmods.covar_module.outputscale  # corresponds to the signal variance (sigma_f^2)
    lengthscale = sampmods.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise = sampmods.likelihood.noise  # noise variance (\sigma_n^2)

    hyperparsamps = [[outputscale[i].reshape(1, 1), lengthscale[i].reshape(dimx, 1), noise[i].reshape(1, 1)]
                     for i in range(nmc)]
    hyperparsamps = [torch.cat(samps, 0).reshape(dimx + 2, ) for samps in hyperparsamps]

    outputscale0 = model0.covar_module.outputscale  # corresponds to the signal variance (sigma_f^2)
    lengthscale0 = model0.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise0 = model0.likelihood.noise  # noise variance (\sigma_n^2)

    theta0 = [outputscale0.reshape(1, 1), lengthscale0.reshape(dimx, 1), noise0.reshape(1, 1)]
    theta0 = torch.cat(theta0, 0).reshape(dimx + 2, )

    conf = 1 - delta_max
    dimpar = theta0.shape[0]

    indmax = round(len(hyperparsamps) * conf)
    inds = torch.as_tensor([torch.abs(samp - theta0).max() for samp in hyperparsamps]).argsort()[:indmax]
    sampsinregion = [hyperparsamps[ind] for ind in inds[:indmax]]
    thprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).min() for i in range(dimpar)])
    thdoubprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).max() for i in range(dimpar)])
    thprimnew = torch.min(thprim, theta0)
    thdoubprimnew = torch.max(thdoubprim, theta0)

    maxsqrbeta = 1.414
    gamma = torch.sqrt(torch.prod(torch.divide(thdoubprimnew[1:-1], thprimnew[1:-1])))
    gamma /= torch.sqrt(torch.prod(torch.divide(thdoubprimnew[0], theta0[0])))
    zeta = 0.1
    betabar = gamma ** 2 * (maxsqrbeta + zeta)
    beta = torch.as_tensor(min(4 * maxsqrbeta ** 2, betabar))
    sqrbeta = torch.sqrt(beta)

    # Create robust bounding hyperparameters. These correspond to the minimal lengthscales
    # and maximal signal/noise variances. Referred to as theta' in the experimental sectoin of the paper
    throbust = thprimnew  # deepcopy(thprimnew)
    throbust[0] = thdoubprimnew[0]  # deepcopy(thdoubprimnew[0])
    throbust[-1] = thdoubprimnew[-1]  # deepcopy(thdoubprimnew[-1])

    robustmodel = deepcopy(model0)
    robustmodel.covar_module.base_kernel._set_lengthscale(throbust[1:-1])
    robustmodel.covar_module._set_outputscale(throbust[0])
    robustmodel.likelihood.noise = throbust[-1]

    return robustmodel, sqrbeta, gamma


def getboundinggp_sm(sampmods, model0, nmc, delta_max):
    # get bounding gp in the case of a spectral mixture kernel

    # set number of MCMC samples and delta if not available
    if not nmc:
        nmc = sampmods.covar_module.mixture_scales.shape[0]
    if not delta_max:
        delta_max = 0.05

    # extract input dimension from lengthscales
    dimx = sampmods.covar_module.mixture_scales.shape[-1]
    num_mixtures = model0.covar_module.num_mixtures

    # concatenate hyperparameter samples generated with NUTS
    weights = sampmods.covar_module.mixture_weights  # corresponds to the signal variance (sigma_f^2)
    lengthscale = [lens.reshape(num_mixtures, ).detach() for lens in
                   sampmods.covar_module.mixture_scales]  # lengthscale SQUARED
    noise = sampmods.likelihood.noise  # noise variance (\sigma_n^2)

    hyperparsamps = [[weights[i].reshape(num_mixtures, 1), lengthscale[i].reshape(dimx * num_mixtures, 1),
                      noise[i].reshape(1, 1)] for i in range(nmc)]
    hyperparsamps = [torch.cat(samps, 0).reshape((1 + dimx) * num_mixtures + 1, ) for samps in hyperparsamps]

    weights0 = model0.covar_module.mixture_weights  # corresponds to the signal variance (sigma_f^2)
    lengthscale0 = model0.covar_module.mixture_scales  # lengthscale (NOT logarithm or square)
    noise0 = model0.likelihood.noise  # noise variance (\sigma_n^2)

    theta0 = [weights0.reshape(num_mixtures, 1), lengthscale0.reshape(dimx * num_mixtures, 1), noise0.reshape(1, 1)]
    theta0 = torch.cat(theta0, 0).reshape((1 + dimx) * num_mixtures + 1, )

    conf = 1 - delta_max

    indmax = round(len(hyperparsamps) * conf)
    inds = torch.as_tensor([torch.abs(samp - theta0).max() for samp in hyperparsamps]).argsort()[:indmax]
    sampsinregion = [hyperparsamps[ind] for ind in inds[:indmax]]
    thprim = torch.tensor([torch.tensor([samp[i]
                                         for samp in sampsinregion]).min() for i in
                           range((1 + dimx) * num_mixtures + 1)])
    thdoubprim = torch.tensor([torch.tensor([samp[i] for
                                             samp in sampsinregion]).max() for i in
                               range((1 + dimx) * num_mixtures + 1)])

    # make sure bounding hyperparameters contain theta0
    thprim = torch.min(thprim, theta0)
    thdoubprim = torch.max(thdoubprim, theta0)

    maxsqrbeta = 1.414
    gamma = torch.sqrt(torch.prod(torch.divide(thdoubprim[num_mixtures:-1], thprim[num_mixtures:-1])))
    gamma /= torch.sqrt(torch.prod(torch.divide(thdoubprim[0], theta0[0])))
    zeta = 0.1
    betabar = gamma ** 2 * (maxsqrbeta + zeta)
    beta = torch.as_tensor(min(4 * maxsqrbeta ** 2, betabar))
    sqrbeta = torch.sqrt(beta)

    # Create robust bounding hyperparameters. These correspond to the minimal lengthscales
    # and maximal signal/noise variances. Referred to as theta' in the experimental sectoin of the paper    # # Create robust bounding hyperparameters
    throbust = thprim  # deepcopy(thprim)
    throbust[:num_mixtures] = thdoubprim[:num_mixtures]  # deepcopy(thdoubprim[0])
    throbust[-1] = thdoubprim[-1]  # deepcopy(thdoubprim[-1])

    robustmodel = deepcopy(model0)
    robustmodel.covar_module._set_mixture_weights(throbust[:num_mixtures])
    robustmodel.covar_module._set_mixture_scales(throbust[num_mixtures:-1])
    robustmodel.likelihood.noise = throbust[-1]

    return robustmodel, sqrbeta, gamma

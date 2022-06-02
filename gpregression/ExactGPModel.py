import torch
import gpytorch
from gpytorch.priors import UniformPrior

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dimx, lb, ub, kernel=False ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if not kernel==False:
            if kernel._get_name()=='SpectralMixtureKernel':
                kernel.initialize_from_data(train_x, train_y)
        else:

            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                           lengthscale_prior=UniformPrior(lb[1:-1],
                                                                          ub[1:-1]),
                                           lengthscale_constraint=gpytorch.constraints.Interval(
                                               lb[1:-1], ub[1:-1])),
                outputscale_prior=UniformPrior(lb[0], ub[0]),
                outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0])
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


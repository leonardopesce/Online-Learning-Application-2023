import gpytorch
import torch

from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood


class BaseGaussianProcess(gpytorch.models.ExactGP):
    """Gausian Process Regressor. Inherits from gpytorch.models.ExactGP.

    Represents a basic Gaussian Process Regressor which relies on PyTorch. 

    Params:
    -------
        :param train_x: Tensor of training data, defaults to torch.Tensor([])
        :type train_x: torch.Tensor, optional
        :param train_y: Tensor of training targets, defaults to torch.Tensor([])
        :type train_y: torch.Tensor, optional
        :param likelihood: Likelihood function, defaults to gpytorch.likelihoods.GaussianLikelihood()
        :type likelihood: gpytorch.likelihoods.GaussianLikelihood, optional
        :param kernel: Kernel function, defaults to gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        :type kernel: gpytorch.kernels.Kernel, optional
    """
    def __init__(self,
                 train_x : torch.Tensor = torch.Tensor([]),
                 train_y : torch.Tensor = torch.Tensor([]),
                 likelihood : gpytorch.likelihoods.GaussianLikelihood = gpytorch.likelihoods.GaussianLikelihood(),
                 kernel : gpytorch.kernels.Kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                 ):
        super(BaseGaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, x_train : torch.Tensor, y_train : torch.Tensor, n_restarts_optimizer : int = 10) -> None:
        """Fit the Gaussian Process Regressor with the given training data.

        Args:
        -----
            :param torch.Tensor x_train: Tensor of training data.
            :param torch.Tensor y_train: Tensor of training targets.
            :param n_restarts_optimizer: Number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood, defaults to 10
            :type n_restarts_optimizer: int, optional
        """
        self.set_train_data(x_train, y_train, strict=False)

        self.train()
        self.likelihood.train()

        optimizer = Adam(self.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for _ in range(n_restarts_optimizer):
            optimizer.zero_grad()
            output = self(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
    
    def predict(self, x_test : torch.Tensor, return_std : bool = True, return_bounds : bool = True) -> tuple:
        """Make predictions with the Gaussian Process Regressor.

        Optionally returns the standard deviation and the lower and upper bounds of the confidence interval.

        Args:
        -----
            :param torch.Tensor x_test: Tensor of data to use for prediction.
            :param return_std: If True, returns the standard deviation of the predictions, defaults to True
            :type return_std: bool, optional
            :param return_bounds: If True, returns the lower and upper bounds of the confidence interval, defaults to True
            :type return_bounds: bool, optional

        Returns:
        --------
            :return: Tuple of the form (mean, std, lower_bound, upper_bound) where:
                - mean: mean of the predictions.
                - std: standard deviation of the predictions. Returned only if return_std is True.
                - lower_bound: lower bound of the confidence interval. Returned only if return_bounds is True.
                - upper_bound: upper bound of the confidence interval. Returned only if return_bounds is True.
            :rtype: tuple
        """
        # Switching to evaluation mode.
        self.eval()
        self.likelihood.eval()

        # Making predictions.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x_test))
            if return_std and return_bounds:
                return observed_pred.mean.numpy(), observed_pred.variance.numpy(), observed_pred.confidence_region()[0].numpy(), observed_pred.confidence_region()[1].numpy()
            elif return_std:
                return observed_pred.mean.numpy(), observed_pred.variance.numpy()
            elif return_bounds:
                return observed_pred.mean.numpy(), observed_pred.confidence_region()[0].numpy(), observed_pred.confidence_region()[1].numpy()
            else:
                return observed_pred.mean.numpy()
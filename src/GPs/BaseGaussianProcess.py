import gpytorch
import torch

from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood

class BaseGaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(BaseGaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, x_train, y_train, n_restarts_optimizer=10):
        # Set new data to the model
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
    
    def predict(self, x_test):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x_test))
            return observed_pred.mean.numpy(), observed_pred.variance.numpy(), observed_pred.confidence_region()[0].numpy(), observed_pred.confidence_region()[1].numpy()
import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Product, ConstantKernel
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import NormalPrior

from Learner import Learner
from GPs import BaseGaussianProcess

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
class GPTS_Learner(Learner):
    """Gaussian Process Thompson Sampling Learner. Inherits from Learner.

    Parameters:
    -----------
        :param np.array arms: List of arms to be pulled. In this case the arms are the bids.
        :param np.array means_clicks: Means of the Gaussian distributions of the clicks.
        :param np.array means_costs: Means of the Gaussian distributions of the costs.
        :param np.array sigmas_clicks: Standard deviations of the Gaussian distributions of the clicks.
        :param np.array sigmas_costs: Standard deviations of the Gaussian distributions of the costs.
        :param np.array lower_bounds_clicks: Lower bounds of the Gaussian distributions of the clicks.
        :param np.array lower_bounds_costs: Lower bounds of the Gaussian distributions of the costs.
        :param np.array upper_bounds_clicks: Upper bounds of the Gaussian distributions of the clicks.
        :param np.array upper_bounds_costs: Upper bounds of the Gaussian distributions of the costs.
        :param list pulled_bids: List of pulled bids.
        :param np.array collected_clicks: List of collected clicks.
        :param np.array collected_costs: List of collected costs.
        :param BaseGaussianProcess gp_clicks: Gaussian Process Regressor for the clicks curve.
        :param BaseGaussianProcess gp_costs: Gaussian Process Regressor for the costs curve.
    """

    def __init__(self, arms, sklearn=True):
        super().__init__(arms)
        self.arms = arms
        self.means_clicks = np.zeros(self.n_arms)
        self.sigmas_clicks = np.ones(self.n_arms)*10
        self.lower_bounds_clicks = np.zeros(self.n_arms)
        self.upper_bounds_clicks = np.zeros(self.n_arms)
        self.means_costs = np.zeros(self.n_arms)
        self.sigmas_costs = np.ones(self.n_arms) * 10
        self.lower_bounds_costs = np.zeros(self.n_arms)
        self.upper_bounds_costs = np.zeros(self.n_arms)
        self.pulled_bids = []
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])
        self.sklearn = sklearn

        if sklearn:
            kernel_clicks = ConstantKernel() * RBF() #+ WhiteKernel() # ScaleKernel(RBFKernel())
            kernel_costs = ConstantKernel() * RBF() #+ WhiteKernel() # ScaleKernel(RBFKernel())
            self.gp_clicks = GaussianProcessRegressor(kernel=kernel_clicks, alpha=1000, n_restarts_optimizer=10)
            self.gp_costs = GaussianProcessRegressor(kernel=kernel_costs, alpha=300, n_restarts_optimizer=10)
        else:
            kernel_clicks = ScaleKernel(RBFKernel())
            kernel_costs = ScaleKernel(RBFKernel())
            likelihood_clicks = GaussianLikelihood(noise_prior=NormalPrior(0, 1000))
            likelihood_costs = GaussianLikelihood(noise_prior=NormalPrior(0, 300))
            self.gp_clicks = BaseGaussianProcess(likelihood=likelihood_clicks, kernel=kernel_clicks)
            self.gp_costs = BaseGaussianProcess(likelihood=likelihood_costs, kernel=kernel_costs)

    def update_observations(self, pulled_arm : int, reward) -> None:
        """Update the reward, number of clicks and cumulative costs after having pulled the selected arm.

        Args:
        -----
            :param int pulled_arm: index of the pulled bid.
            :param tuple reward: tuple of the form (reward, n_clicks, costs) sampled from the environment.
        """
        # Here reward is a tuple:
        # reward[0] = reward of the environment
        # reward[1] = n_clicks sampled from the environment
        # reward[2] = costs sampled from the environment
        super().update_observations(pulled_arm, reward[0])
        self.pulled_bids.append(self.arms[pulled_arm])

        # Save the clicks and comulative costs for the pulled bid
        self.collected_clicks = np.append(self.collected_clicks, reward[1])
        self.collected_costs = np.append(self.collected_costs, reward[2])

    def update_model(self) -> None:
        """Updates the means and standard deviations of the Gaussian distributions of the clicks and costs curves fitting a Gaussian process model."""
        if self.sklearn:
            x = np.array(self.pulled_bids).reshape(-1 ,1) # Bids previously pulled.
            y = np.array(self.collected_clicks).reshape(-1 ,1) # torch.Tensor(self.collected_clicks)       # Clicks previously collected.

            # Fitting the Gaussian Process Regressor relative to clicks and making a prediction for the current round.
            self.gp_clicks.fit(x, y)
            self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(self.arms.reshape(-1 ,1), return_std=True) # , self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.arms))

            # Fitting the Gaussian Process Regressor relative to costs and making a prediction for the current round.
            y = np.array(self.collected_costs).reshape(-1 ,1) # torch.Tensor(self.collected_costs)        # Daily costs previously collected.
            self.gp_costs.fit(x, y)
            self.means_costs, self.sigmas_costs = self.gp_costs.predict(self.arms.reshape(-1 ,1), return_std=True) # , self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.arms))
        else:
            x = torch.Tensor(self.pulled_bids) # Bids previously pulled.
            y = torch.Tensor(self.collected_clicks) # Clicks previously collected.

            # Fitting the Gaussian Process Regressor relative to clicks and making a prediction for the current round.
            self.gp_clicks.fit(x, y)
            self.means_clicks, self.sigmas_clicks, self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.arms))
            self.sigmas_clicks = np.sqrt(self.sigmas_clicks)

            # Fitting the Gaussian Process Regressor relative to costs and making a prediction for the current round.
            y = torch.Tensor(self.collected_costs)        # Daily costs previously collected.
            self.gp_costs.fit(x, y)
            self.means_costs, self.sigmas_costs, self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.arms))
            self.sigmas_costs = np.sqrt(self.sigmas_costs)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)

    def update(self, pulled_arm : int, reward) -> None:
        """Updates the timestep, the observations and the model of the thompson sampling algorithm.

        Args:
        -----
            :param int pulled_arm: index of the pulled bid
            :param tuple reward: tuple of the form (reward, n_clicks, costs) sampled from the environment.
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm_GPs(self, prob_margin : float) -> int:
        """Decides which arm to pull based on the current estimations of the clicks and cumulative daily costs.

        Args:
        -----
            :param float prob_margin: conversion_rate * (price - other_costs)

        Returns:
        --------
            :return: index of the bid to pull in the current round.
            :rtype: int
        """
        # Sampling from a normal distribution with mean and std estimated by the GP
        # Do the same for clicks and costs.
        sampled_values_clicks = np.random.normal(self.means_clicks, self.sigmas_clicks)
        sampled_values_costs = np.random.normal(self.means_costs, self.sigmas_costs)

        # Compute the reward got for each bid given the sampled values.
        rewards = sampled_values_clicks * prob_margin - sampled_values_costs

        # Pulling the bid that maximizes the reward
        bid_idx = np.argmax(rewards)

        # Return the index of the pulled bid.
        return bid_idx
    
    def plot_clicks(self) -> None:
        """Plot the clicks curve and the confidence interval together with the data points."""
        plt.figure(0)
        plt.scatter(self.pulled_bids, self.collected_clicks, color='r', label = 'clicks per bid')
        plt.plot(self.arms, self.means_clicks, color='r', label = 'mean clicks')
        plt.fill(np.concatenate([self.arms, self.arms[::-1]]),
                 np.concatenate([self.means_clicks - 1.96 * self.sigmas_clicks,
                                 (self.means_clicks + 1.96 * self.sigmas_clicks)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        # plt.fill_between(self.arms, self.means_clicks - 1.96 * self.sigmas_clicks, self.means_clicks + 1.96 * self.sigmas_clicks, alpha=0.2, color='r')
        plt.title('Clicks TS')
        plt.legend()
        plt.show()

    def plot_costs(self) -> None:
        """Plot the costs curve and the confidence interval together with the data points."""
        plt.figure(1)
        plt.scatter(self.pulled_bids, self.collected_costs, color='b', label = 'costs per bid')
        plt.plot(self.arms, self.means_costs, color='b', label = 'mean costs')
        plt.fill_between(self.arms, self.lower_bounds_costs, self.upper_bounds_costs, alpha=0.2, color='b')
        plt.legend()
        plt.show()

    def sample_clicks(self):
        """
        Sample from a normal distribution with mean and standard deviation estimated by the GP of the number of clicks

        :return: Estimate number of clicks for all the bids
        :rtype: float
        """

        return np.random.normal(self.means_clicks, self.sigmas_clicks)

    def sample_costs(self):
        """
        Sample from a normal distribution with mean and standard deviation estimated by the GP of the costs of the clicks

        :return: Estimate costs of the clicks for all the bids
        :rtype: float
        """

        return np.random.normal(self.means_costs, self.sigmas_costs)

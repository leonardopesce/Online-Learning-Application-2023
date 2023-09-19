import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import NormalPrior
from .Learner import Learner
from src.Utilities.GPs import BaseGaussianProcess

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class GPUCB_Learner(Learner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) algorithm

    :param np.ndarray arms_values: Values associated to the arms
    :param np.ndarray means_clicks: Empirical means of the clicks
    :param np.ndarray confidence_clicks: Confidence intervals of the clicks
    :param np.ndarray means_costs: Empirical means of the costs
    :param np.ndarray confidence_costs: Confidence intervals of the costs
    :param np.ndarray sigmas_clicks: Standard deviations of the clicks
    :param np.ndarray lower_bounds_clicks: Lower bounds of the clicks
    :param np.ndarray upper_bounds_clicks: Upper bounds of the clicks
    :param np.ndarray sigmas_costs: Standard deviations of the costs
    :param np.ndarray lower_bounds_costs: Lower bounds of the costs
    :param np.ndarray upper_bounds_costs: Upper bounds of the costs
    :param list pulled_bids: List of the pulled bids values
    :param np.ndarray collected_clicks: Array of the collected clicks values
    :param np.ndarray collected_costs: Array of the collected costs values
    :param confidence_level: Confidence level of the algorithm regret bounds, defaults to 0.95
    :type confidence_level: float, optional
    :param delta: Delta value of the algorithm regret bounds. The bounds hold in high probability (1 - delta), defaults to 0.05
    :type delta: float, optional
    :param BaseGaussianProcess gp_clicks: Gaussian Process Regressor for the clicks
    :param BaseGaussianProcess gp_costs: Gaussian Process Regressor for the costs
    """

    def __init__(self, arms_values, confidence_level=0.95, sklearn=False):
        super().__init__(arms_values)
        self.means_clicks = np.zeros(self.n_arms)
        self.confidence_clicks = np.array([np.inf] * self.n_arms)
        self.means_costs = np.zeros(self.n_arms)
        self.confidence_costs = np.array([np.inf] * self.n_arms)

        self.sigmas_clicks = np.ones(self.n_arms) * np.sqrt(10)
        self.lower_bounds_clicks = np.zeros(self.n_arms)
        self.upper_bounds_clicks = np.zeros(self.n_arms)
        self.sigmas_costs = np.ones(self.n_arms) * np.sqrt(10)
        self.lower_bounds_costs = np.zeros(self.n_arms)
        self.upper_bounds_costs = np.zeros(self.n_arms)
        self.pulled_bids = []
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])
        self.confidence_level = confidence_level
        self.delta = 1 - confidence_level
        self.sklearn = sklearn

        if sklearn:
            kernel_clicks = ConstantKernel() * RBF() + WhiteKernel() # Product(ConstantKernel(), RBF()) + WhiteKernel()
            kernel_costs = ConstantKernel() * RBF() + WhiteKernel() # Product(ConstantKernel(), RBF()) + WhiteKernel()
            self.gp_clicks = GaussianProcessRegressor(kernel=kernel_clicks, alpha=5, n_restarts_optimizer=5)
            self.gp_costs = GaussianProcessRegressor(kernel=kernel_costs, alpha=5, n_restarts_optimizer=5)
        else:
            kernel_clicks = ScaleKernel(RBFKernel())
            kernel_costs = ScaleKernel(RBFKernel())
            likelihood_clicks = GaussianLikelihood(noise_prior=NormalPrior(0, 50))
            likelihood_costs = GaussianLikelihood(noise_prior=NormalPrior(0, 100))
            #likelihood_clicks = GaussianLikelihood()
            #likelihood_costs = GaussianLikelihood()
            self.gp_clicks = BaseGaussianProcess(likelihood=likelihood_clicks, kernel=kernel_clicks)
            self.gp_costs = BaseGaussianProcess(likelihood=likelihood_costs, kernel=kernel_costs)

    def update_model(self) -> None:
        """Updates the means and standard deviations of the Gaussian distributions of the clicks and costs curves fitting a Gaussian process model."""
        if self.sklearn:
            x = np.array(self.pulled_bids).reshape(-1, 1) # torch.Tensor(self.pulled_bids)            # Bids previously pulled.

            # Fitting the Gaussian Process Regressor relative to clicks and making a prediction for the current round.
            y = self.collected_clicks.reshape(-1, 1) # torch.Tensor(self.collected_clicks)       # Clicks previously collected.
            self.gp_clicks.fit(x, y)
            self.empirical_means_clicks, self.sigmas_clicks = self.gp_clicks.predict(self.get_arms().reshape(-1 ,1), return_std=True)
        
            beta = 2 * np.log((self.n_arms * (self.t ** 2) * (np.pi ** 2)) / (6 * self.delta)) # https://arxiv.org/pdf/0912.3995.pdf
            self.confidence_clicks = np.sqrt(beta) * self.sigmas_clicks

            # Fitting the Gaussian Process Regressor relative to costs and making a prediction for the current round.
            y = self.collected_costs.reshape(-1, 1)        # Daily costs previously collected.
            self.gp_costs.fit(x, y)
            self.empirical_means_costs, self.sigmas_costs = self.gp_costs.predict(self.get_arms().reshape(-1 ,1), return_std=True)

            self.confidence_costs = np.sqrt(beta) * self.sigmas_costs
        else:
            x = torch.Tensor(self.pulled_bids)            # Bids previously pulled.

            # Fitting the Gaussian Process Regressor relative to clicks and making a prediction for the current round.
            y = torch.Tensor(self.collected_clicks)       # Clicks previously collected.
            self.gp_clicks.fit(x, y)
            self.means_clicks, self.sigmas_clicks, self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.arms_values))
            self.sigmas_clicks = np.sqrt(self.sigmas_clicks)

            beta = 2 * np.log((self.n_arms * (self.t ** 2) * (np.pi ** 2)) / (6 * self.delta))  # https://arxiv.org/pdf/0912.3995.pdf
            self.confidence_clicks = np.sqrt(beta) * self.sigmas_clicks

            # Fitting the Gaussian Process Regressor relative to costs and making a prediction for the current round.
            y = torch.Tensor(self.collected_costs)        # Daily costs previously collected.
            self.gp_costs.fit(x, y)
            self.means_costs, self.sigmas_costs, self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.get_arms()))
            self.sigmas_costs = np.sqrt(self.sigmas_costs)

            self.confidence_costs = np.sqrt(beta) * self.sigmas_costs
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)

    def pull_arm_GPs(self, prob_margin : float) -> int:
        """
        Chooses the arm to play based on the UCB1 algorithm, therefore choosing the arm with higher upper
        confidence bound, which is the mean of the reward of the arm plus the confidence interval
        
        :param float prob_margin: conversion_rate * (price - other_costs)
        :return: Index of the arm to pull
        :rtype: int
        """

        upper_confidence_bound_clicks = self.means_clicks + self.confidence_clicks
        lower_confidence_bound_costs = self.means_costs - self.confidence_costs
        reward = upper_confidence_bound_clicks * prob_margin - lower_confidence_bound_costs
        idx = np.random.choice(np.where(reward == reward.max())[0])

        return idx

    def update_observations(self, pulled_arm : int, reward) -> None:
        """
        Update the reward, number of clicks and cumulative costs after having pulled the selected arm

        :param int pulled_arm: index of the pulled bid
        :param tuple reward: tuple of the form (reward, n_clicks, costs) sampled from the environment
        """
        # Here reward is a tuple:
        # reward[0] = reward of the environment
        # reward[1] = n_clicks sampled from the environment
        # reward[2] = costs sampled from the environment
        super().update_observations(pulled_arm, reward[0])
        self.pulled_bids.append(self.get_arms()[pulled_arm])

        # Save the clicks and comulative costs for the pulled bid
        self.collected_clicks = np.append(self.collected_clicks, reward[1])
        self.collected_costs = np.append(self.collected_costs, reward[2])

    def update(self, pulled_arm : int, reward, model_update = True) -> None:
        """
        Updating the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param tuple reward: tuple of the form (reward, n_clicks, costs) sampled from the environment
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)

        if model_update:
            self.update_model()

    def get_confidence_bounds(self):
        """
        Returns the upper confidence bound for all the arms

        Returns:
            upper confidence bound for all the arms
        """
        return self.means_clicks + self.confidence_clicks, self.means_costs - self.confidence_costs

    def plot_clicks(self) -> None:
        """Plot the clicks curve and the confidence interval together with the data points."""
        plt.figure(0)
        plt.scatter(self.pulled_bids, self.collected_clicks, color='r', label = 'clicks per bid')
        plt.plot(self.arms_values, self.empirical_means_clicks, color='r', label = 'mean clicks')
        # plt.fill_between(self.arms_values, self.empirical_means_clicks - self.sigmas_clicks, self.empirical_means_clicks + self.sigmas_clicks, alpha=0.2, color='r')
        plt.fill(np.concatenate([self.arms_values, self.arms_values[::-1]]),
                 np.concatenate([self.empirical_means_clicks - 1.96 * self.sigmas_clicks,
                                 (self.empirical_means_clicks + 1.96 * self.sigmas_clicks)[::-1]]),
                 alpha=.3, fc='orange', ec='None', label='95% confidence interval')
        plt.title('Clicks UCB')
        plt.legend()
        plt.show()
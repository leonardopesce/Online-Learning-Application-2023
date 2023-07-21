import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

from Learner import *
from Environment import fun
from GPs import BaseGaussianProcess

class GPTS_Learner(Learner):
    """Gaussian Process Thompson Sampling Learner. Inherits from Learner.

    Parameters:
    -----------
        :param np.array arms: List of arms to be pulled. In this case the arms are the bids.
        :param np.array means_clicks: Means of the Gaussian distributions of the clicks.
        :param np.array means_costs: Means of the Gaussian distributions of the costs.
        :param np.array sigmas_clicks: Standard deviations of the Gaussian distributions of the clicks.
        :param np.array sigmas_costs: Standard deviations of the Gaussian distributions of the costs.
        :param list pulled_bids: List of pulled bids.
        :param np.array collected_clicks: List of collected clicks.
        :param np.array collected_costs: List of collected costs.
        :param GaussianProcessRegressor gp_clicks: Gaussian Process Regressor for the clicks curve.
        :param GaussianProcessRegressor gp_costs: Gaussian Process Regressor for the costs curve.
    """

    def __init__(self, arms):
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

        # kernel_clicks = C(1.0) * RBF(1.0) + W(1e-1)
        # kernel_costs = C(1.0) * RBF(1.0) + W(1e-1)
        kernel_clicks = ScaleKernel(RBFKernel())
        kernel_costs = ScaleKernel(RBFKernel())
        likelihood_clicks = GaussianLikelihood()
        likelihood_costs = GaussianLikelihood()
        
        # self.gp_clicks = GaussianProcessRegressor(kernel = kernel_clicks, n_restarts_optimizer = 10)
        # self.gp_costs = GaussianProcessRegressor(kernel = kernel_costs, n_restarts_optimizer = 10)
        self.gp_clicks = BaseGaussianProcess(train_x=torch.Tensor(self.pulled_bids), train_y=torch.Tensor(self.collected_clicks), likelihood=likelihood_clicks, kernel=kernel_clicks)
        self.gp_costs = BaseGaussianProcess(train_x=torch.Tensor(self.pulled_bids), train_y=torch.Tensor(self.collected_costs), likelihood=likelihood_costs, kernel=kernel_costs)

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
        # x = np.atleast_2d(self.pulled_bids).T       # Bids previously pulled
        x = torch.Tensor(self.pulled_bids)
        # y = self.collected_clicks                   # Clicks previously collected.
        y = torch.Tensor(self.collected_clicks)
        self.gp_clicks.fit(x, y)
        # self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means_clicks, self.sigmas_clicks, self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.arms))
        self.sigmas_clicks = np.sqrt(self.sigmas_clicks)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)

        # y = self.collected_costs                    # Daily costs previously collected.
        y = torch.Tensor(self.collected_costs)
        self.gp_costs.fit(x, y)
        # self.means_costs, self.sigmas_costs = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means_costs, self.sigmas_costs, self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.arms))
        self.sigmas_costs = np.sqrt(self.sigmas_costs)
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
        plt.figure(0)
        plt.scatter(self.pulled_bids, self.collected_clicks, color='r', label = 'clicks per bid')
        plt.plot(self.arms, self.means_clicks, color='r', label = 'mean clicks')
        # plt.fill_between(self.arms, self.means_clicks - self.sigmas_clicks, self.means_clicks + self.sigmas_clicks, alpha=0.2, color='r')
        plt.fill_between(self.arms, self.lower_bounds_clicks, self.upper_bounds_clicks, alpha=0.2, color='r')
        plt.legend()
        plt.show()

    def plot_costs(self) -> None:
        plt.figure(1)
        plt.scatter(self.pulled_bids, self.collected_costs, color='b', label = 'costs per bid')
        plt.plot(self.arms, self.means_costs, color='b', label = 'mean costs')
        # plt.fill_between(self.arms, self.means_costs - self.sigmas_costs, self.means_costs + self.sigmas_costs, alpha=0.2, color='b')
        plt.fill_between(self.arms, self.lower_bounds_costs, self.upper_bounds_costs, alpha=0.2, color='b') 
        plt.legend()
        plt.show()
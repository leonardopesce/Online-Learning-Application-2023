from Learner import *
import numpy as np
import matplotlib.pyplot as plt
import torch

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from Learner import Learner
from GPs import BaseGaussianProcess


class GPUCB_Learner(Learner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) algorithm

    :param np.ndarray arms_values: Values associated to the arms
    empirical_means: Empirical means
    confidence: upper Confidence interval
    """

    def __init__(self, arms_values):
        super().__init__(arms_values)
        self.empirical_means_clicks = np.zeros(self.n_arms)
        self.confidence_clicks = np.array([np.inf] * self.n_arms)
        self.empirical_means_costs = np.zeros(self.n_arms)
        self.confidence_costs = np.array([np.inf] * self.n_arms)

        self.sigmas_clicks = np.ones(self.n_arms) * 10
        self.lower_bounds_clicks = np.zeros(self.n_arms)
        self.upper_bounds_clicks = np.zeros(self.n_arms)
        self.sigmas_costs = np.ones(self.n_arms) * 10
        self.lower_bounds_costs = np.zeros(self.n_arms)
        self.upper_bounds_costs = np.zeros(self.n_arms)
        self.pulled_bids = []
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])

        kernel_clicks = ScaleKernel(RBFKernel())
        kernel_costs = ScaleKernel(RBFKernel())
        likelihood_clicks = GaussianLikelihood()
        likelihood_costs = GaussianLikelihood()

        self.gp_clicks = BaseGaussianProcess(likelihood=likelihood_clicks, kernel=kernel_clicks)
        self.gp_costs = BaseGaussianProcess(likelihood=likelihood_costs, kernel=kernel_costs)

    def update_model(self) -> None:
        """Updates the means and standard deviations of the Gaussian distributions of the clicks and costs curves fitting a Gaussian process model."""
        x = torch.Tensor(self.pulled_bids)            # Bids previously pulled.

        # Fitting the Gaussian Process Regressor relative to clicks and making a prediction for the current round.
        y = torch.Tensor(self.collected_clicks)       # Clicks previously collected.
        self.gp_clicks.fit(x, y)
        self.empirical_means_clicks, self.sigmas_clicks, self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.arms_values))
        self.sigmas_clicks = np.sqrt(self.sigmas_clicks)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)
        '''
        for arm in range(self.n_arms):
            self.confidence_clicks[arm] = np.sqrt((2 * np.log(self.t) / self.times_arms_played[arm])) if self.times_arms_played[arm] > 0 else 50
        self.confidence_clicks = self.confidence_clicks * self.sigmas_clicks'''
        self.confidence_clicks = np.sqrt(10) * self.sigmas_clicks

        # Fitting the Gaussian Process Regressor relative to costs and making a prediction for the current round.
        y = torch.Tensor(self.collected_costs)        # Daily costs previously collected.
        self.gp_costs.fit(x, y)
        self.empirical_means_costs, self.sigmas_costs, self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.arms))
        self.sigmas_costs = np.sqrt(self.sigmas_costs)
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)
        '''
        for arm in range(self.n_arms):
            self.confidence_costs[arm] = np.sqrt((2 * np.log(self.t) / self.times_arms_played[arm])) if self.times_arms_played[arm] > 0 else 10
        self.confidence_costs = self.confidence_costs * self.sigmas_costs'''
        self.confidence_costs = np.sqrt(10) * self.sigmas_costs

    def pull_arm_GPs(self, prob_margin) -> int:
        """
        Chooses the arm to play based on the UCB1 algorithm, therefore choosing the arm with higher upper
        confidence bound, which is the mean of the reward of the arm plus the confidence interval

        :return: Index of the arm to pull
        :rtype: int
        """

        upper_confidence_bound_clicks = self.empirical_means_clicks + self.confidence_clicks
        lower_confidence_bound_costs = self.empirical_means_costs - self.confidence_costs
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
        self.pulled_bids.append(self.arms[pulled_arm])

        # Save the clicks and comulative costs for the pulled bid
        self.collected_clicks = np.append(self.collected_clicks, reward[1])
        self.collected_costs = np.append(self.collected_costs, reward[2])

    def update(self, pulled_arm : int, reward) -> None:
        """
        Updating the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param tuple reward: tuple of the form (reward, n_clicks, costs) sampled from the environment
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
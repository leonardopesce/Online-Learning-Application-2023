from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

class GPTS_Learner(Learner):

    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means_clicks = np.zeros(self.n_arms)
        self.sigmas_clicks = np.ones(self.n_arms)*10
        self.means_costs = np.zeros(self.n_arms)
        self.sigmas_costs = np.ones(self.n_arms) * 10
        self.pulled_bids = []
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])

        alpha_clicks = 0.5
        alpha_costs = 0.5
        kernel_clicks = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))
        kernel_costs = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))
        self.gp_clicks = GaussianProcessRegressor(kernel = kernel_clicks, alpha = alpha_clicks ** 2, n_restarts_optimizer = 10)
        self.gp_costs = GaussianProcessRegressor(kernel = kernel_costs, alpha = alpha_costs ** 2, n_restarts_optimizer = 10)

    def update_observations(self, pulled_arm, reward):
        # Here reward is a tuple:
        # reward[0] = reward of the environment
        # reward[1] = n_clicks sampled from the environment
        # reward[2] = costs sampled from the environment
        super().update_observations(pulled_arm, reward[0])
        self.pulled_bids.append(self.arms[pulled_arm])

        # Save the clicks and comulative costs for the pulled bid
        self.collected_clicks = np.append(self.collected_clicks, reward[1])
        self.collected_costs = np.append(self.collected_costs, reward[2])

    def update_model(self):
        x = np.atleast_2d(self.pulled_bids).T
        y = self.collected_clicks
        self.gp_clicks.fit(x, y)
        self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)

        y = self.collected_costs
        self.gp_costs.fit(x, y)
        self.means_costs, self.sigmas_costs = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm_GPs(self, prob_margin):
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
    
    def plot_clicks(self):
        plt.figure(0)
        plt.scatter(self.pulled_bids, self.collected_clicks, color='r', label = 'clicks per bid')
        plt.plot(self.arms, self.means_clicks, color='r', label = 'mean clicks')
        plt.fill_between(self.arms, self.means_clicks - self.sigmas_clicks, self.means_clicks + self.sigmas_clicks, alpha=0.2, color='r')
        plt.legend()
        plt.show()

    def plot_costs(self):
        plt.figure(1)
        plt.scatter(self.pulled_bids, self.collected_costs, color='b', label = 'costs per bid')
        plt.plot(self.arms, self.means_costs, color='b', label = 'mean costs')
        plt.fill_between(self.arms, self.means_costs - self.sigmas_costs, self.means_costs + self.sigmas_costs, alpha=0.2, color='b')        
        plt.legend()
        plt.show()
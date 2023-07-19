from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):

    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means_clicks = np.zeros(self.n_arms)
        self.sigmas_clicks = np.ones(self.n_arms) * 10
        self.means_costs = np.zeros(self.n_arms)
        self.sigmas_costs = np.ones(self.n_arms) * 10
        self.pulled_bids = []
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp_clicks = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)
        self.gp_costs = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_bids.append(self.arms[arm_idx])

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
        clicks = np.random.normal(self.means_clicks, self.sigmas_clicks)
        cum_costs = np.random.normal(self.means_costs, self.sigmas_costs)
        rewards = clicks * prob_margin - cum_costs
        bid_idx = np.argmax(rewards)

        # Save the clicks and comulative costs for the pulled bid
        self.collected_clicks = np.append(self.collected_clicks, clicks[bid_idx])
        self.collected_costs = np.append(self.collected_costs, cum_costs[bid_idx])

        return bid_idx
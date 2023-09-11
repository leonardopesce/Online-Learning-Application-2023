import numpy as np

from PricingAdvertisingLearner import PricingAdvertisingLearner
from GPTS_Learner import GPTS_Learner
from TSReward import TSRewardLearner

# TODO This learner maximize the reward given prices and bids in a joint way. Do we what this?


class TSLearnerPricingAdvertising(PricingAdvertisingLearner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm to the problem of advertising and pricing

    Attributes:
        learner_pricing: Learner that applies the Thompson Sampling(TS) algorithm to the pricing problem
        GP_advertising: Learner that applies the Gaussian Process Thompson Sampling(GPTS) algorithm to the advertising problem
    """

    def __init__(self, prices, bids):
        """
        Initializes the learner

        :params numpy.ndarray prices: Prices in the pricing problem
        :params numpy.ndarray bids: Bids in the advertising problem
        """

        self.learner_pricing = TSRewardLearner(prices)
        self.GP_advertising = GPTS_Learner(bids)

    def pull_arm(self, other_costs):
        """
        Chooses the price and bid to play based on the TS algorithm, therefore it samples the Beta distribution and the
        Gaussian processes of the advertising problem and then it chooses the price and the bid that maximize the reward

        :param float other_costs: Know costs of the product, used to compute the margin

        :return: Index of the price to pull, index of the bid to pull
        :rtype: tuple
        """

        # Sampling the beta distributions in order to have the estimates of the conversion probabilities
        beta_distributions = self.learner_pricing.get_betas()
        sampled_beta_distributions = np.random.beta(beta_distributions[:, 0], beta_distributions[:, 1])
        # Computing the product between the estimates of the conversion probabilities and the margins
        conversion_times_margin = sampled_beta_distributions * (self.learner_pricing.arms_values - other_costs)

        # Sampling from a normal distribution with mean and std estimated by the GP
        sampled_values_clicks = self.GP_advertising.sample_clicks()
        sampled_values_costs = self.GP_advertising.sample_costs()

        # Computing the reward got for each price and bid it is possible to pull.
        rewards = sampled_values_clicks[None, :] * conversion_times_margin[:, None] - sampled_values_costs[None, :]
        # Pulling the bid that maximizes the reward
        flat_index_maximum = np.argmax(rewards)
        num_prices, num_bids = rewards.shape
        best_price_idx = flat_index_maximum // num_bids
        best_bid_idx = flat_index_maximum % num_bids

        return best_price_idx, best_bid_idx

    def update(self, pulled_price, bernoulli_realizations, pulled_bid, n_clicks, costs_adv, reward, model_update = True):
        """
        Updating the parameters of the learners based on the observations obtained by playing the chosen price and bid
        in the environment

        :param int pulled_price: Price pulled in the current time step
        :param bernoulli_realizations: Bernoulli realizations of the pulled price in the current time step
        :param pulled_bid: Bid pulled in the current time step
        :param n_clicks: Number of clicks obtained playing the bid in the current time step
        :param costs_adv: Cost due to the advertising when playing the bid in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        self.learner_pricing.update(pulled_price, reward, bernoulli_realizations)
        self.GP_advertising.update(pulled_bid, (reward, n_clicks, costs_adv), model_update)

    def get_pulled_prices(self):
        """
        Returns the ordered sequence of pulled prices

        :returns: Ordered sequence of pulled prices
        :rtype: list
        """

        return self.learner_pricing.pulled_arms

    def get_pulled_bids(self):
        """
        Returns the ordered sequence of pulled bids

        :returns: Ordered sequence of pulled bids
        :rtype: list
        """

        return self.GP_advertising.pulled_arms

    def get_reward(self):
        return self.learner_pricing.collected_rewards

    @property
    def t(self):
        if self.learner_pricing.t != self.GP_advertising.t:
            raise ValueError("TS and GPTS have different time steps")
        return self.learner_pricing.t

    @t.setter
    def t(self, t):
        self.learner_pricing.t = t
        self.GP_advertising.t = t

    @property
    def advertising_learner(self):
        return self.GP_advertising

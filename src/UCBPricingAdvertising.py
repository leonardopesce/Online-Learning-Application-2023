import numpy as np

from PricingAdvertisingLearner import PricingAdvertisingLearner
from GPUCB_Learner import GPUCB_Learner
from UCB import UCBLearner

# TODO This learner maximize the reward given prices and bids in a joint way


class UCBLearnerPricingAdvertising(PricingAdvertisingLearner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) algorithm to the problem of advertising and pricing

    Attributes:
        learner_pricing: Learner that applies the Upper Confidence Bound 1(UCB1) algorithm to the pricing problem
        GP_advertising: Learner that applies the Upper Confidence Bound 1(UCB1) algorithm to the advertising problem
    """

    def __init__(self, prices, bids, sklearn=True):
        """
        Initializes the learner

        :params numpy.ndarray prices: Prices in the pricing problem
        :params numpy.ndarray bids: Bids in the advertising problem
        """

        self.learner_pricing = UCBLearner(prices)
        self.GP_advertising = GPUCB_Learner(bids, sklearn=sklearn)

    def pull_arm(self, other_costs):
        """
        Chooses the price to play based on the UCB1 algorithm, therefore it computes the upper confidence bounds of the
        conversion probabilities of the prices, the upper confidence bounds of the number of clicks using the Gaussian
        process and the lower confidence bounds of the cumulative daily costs using the Gaussian process, then it
        chooses the price and the bid that maximize the reward

        :param float other_costs: Know costs of the product, used to compute the margin

        :return: Index of the price to pull, index of the bid to pull
        :rtype: tuple
        """

        # Getting the upper confidence bounds from the learner of the price
        prices_upper_conf_bounds = np.clip(self.learner_pricing.get_upper_confidence_bounds(), 0, 1)
        conversion_times_margin = prices_upper_conf_bounds * (self.learner_pricing.get_arms() - other_costs)
        # Getting the upper confidence bounds of the number of clicks from the learner of the bids (advertising problem)
        upper_conf_bounds_clicks, lower_conf_bounds_costs = self.GP_advertising.get_confidence_bounds()

        # Computing the reward got for each price and bid it is possible to pull.
        rewards = upper_conf_bounds_clicks[None, :] * conversion_times_margin[:, None] - lower_conf_bounds_costs[None, :]
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
            raise ValueError("The two learners have different time steps")
        return self.learner_pricing.t

    @t.setter
    def t(self, value):
        self.learner_pricing.t = value
        self.GP_advertising.t = value

    @property
    def advertising_learner(self):
        return self.GP_advertising

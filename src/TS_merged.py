from GPTS_Learner import GPTS_Learner
from Learner import Learner
from TSReward import TSRewardLearner


class TSLearnerMerged(Learner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm to the problem of advertising and pricing

    TS_pricing: Learner that applies the Thompson Sampling(TS) algorithm to the pricing problem
    GPTS_advertising: Learner that applies the Gaussian Process Thompson Sampling(TS) algorithm to the advertising problem
    """

    # TODO maybe prices can be put inside the learner pricing
    def __init__(self, n_prices, prices, n_bids, bids, other_costs):
        """
        Initialize the learner given the environment to solve

        :param int n_prices: Number of prices to use
        :param ndarray prices: List of the prices in the pricing problem
        :param int n_bids: Number of bids to use
        :param ndarray bids: List of the bids in the advertising problem
        :param float other_costs: Know costs of the product, used to compute the margin
        """

        self.prices = prices
        self.TS_pricing = TSRewardLearner(n_prices)
        self.GPTS_advertising = GPTS_Learner(n_bids, bids)
        self.other_costs = other_costs

    def pull_arm(self):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the Beta distribution and choosing the arm
        from whose distribution is extracted the maximum value and then sampling the Gaussian processes of the
        advertising problem and computing the bid that maximize the reward

        :return: Index of the arm to pull, index of the bid to pull
        :rtype: int
        """
        # TODO to evaluate whether to maximize the reward in a joint way in pricing and advertising or not
        best_price_idx = self.TS_pricing.pull_arm(self.prices, self.other_costs)
        conversion_probability_estimate = self.TS_pricing.get_conv_prob(best_price_idx)
        best_bid_idx = self.GPTS_advertising.pull_arm_GPs(conversion_probability_estimate*(self.prices[best_price_idx] - self.other_costs))

        return best_price_idx, best_bid_idx

    def update(self, pulled_price, bernoulli_realization, pulled_bid, n_clicks, costs_adv, reward):
        self.TS_pricing.update(pulled_price, reward, bernoulli_realization)
        self.GPTS_advertising.update(pulled_bid, (reward, n_clicks, costs_adv))

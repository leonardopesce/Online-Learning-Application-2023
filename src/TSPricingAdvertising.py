from GPTS_Learner import GPTS_Learner
from Learner import Learner
from TSReward import TSRewardLearner


class TSLearnerMerged(Learner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm to the problem of advertising and pricing

    TS_pricing: Learner that applies the Thompson Sampling(TS) algorithm to the pricing problem
    GPTS_advertising: Learner that applies the Gaussian Process Thompson Sampling(TS) algorithm to the advertising problem
    other_costs : Know costs of the product, used to compute the margin
    """

    # TODO maybe prices can be put inside the learner pricing
    def __init__(self, prices, bids, other_costs):
        """
        Initialize the learner given the environment to solve

        :param ndarray prices: List of the prices in the pricing problem
        :param ndarray bids: List of the bids in the advertising problem
        :param float other_costs: Know costs of the product, used to compute the margin
        """

        self.TS_pricing = TSRewardLearner(prices)
        self.GPTS_advertising = GPTS_Learner(bids)
        self.other_costs = other_costs

    def pull_arm(self):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the Beta distribution and choosing the arm
        from whose distribution is extracted the maximum value and then sampling the Gaussian processes of the
        advertising problem and computing the bid that maximize the reward

        :return: Index of the arm to pull, index of the bid to pull
        :rtype: tuple
        """

        # TODO to evaluate whether to maximize the reward in a joint way in pricing and advertising or not
        best_price_idx = self.TS_pricing.pull_arm(self.prices, self.other_costs)
        conversion_probability_estimate = self.TS_pricing.get_conv_prob(best_price_idx)
        best_bid_idx = self.GPTS_advertising.pull_arm_GPs(conversion_probability_estimate*(self.prices[best_price_idx] - self.other_costs))

        return best_price_idx, best_bid_idx

    def update(self, pulled_price, bernoulli_realizations, pulled_bid, n_clicks, costs_adv, reward):
        """
        Updating alpha and beta of the beta distribution given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_price: Price pulled in the current time step
        :param bernoulli_realizations: Bernoulli realizations of the pulled price in the current time step
        :param pulled_bid: Bid pulled in the current time step
        :param n_clicks: Number of clicks obtained playing the bid in the current time step
        :param costs_adv: Cost due to the advertising when playing the bid in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        self.TS_pricing.update(pulled_price, reward, bernoulli_realizations)
        self.GPTS_advertising.update(pulled_bid, (reward, n_clicks, costs_adv))


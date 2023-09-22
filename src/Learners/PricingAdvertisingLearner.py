import abc


class PricingAdvertisingLearner(abc.ABC):
    """
    Learner that applies a multi-armed bandit algorithm to the problem of advertising and pricing
    """

    @abc.abstractmethod
    def pull_arm(self, other_costs):
        return

    @abc.abstractmethod
    def update(self, pulled_price, bernoulli_realizations, pulled_bid, n_clicks, costs_adv, reward):
        return

    @abc.abstractmethod
    def get_reward(self):
        return

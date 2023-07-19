from Environment import *


class Clairvoyant:
    """

    """
    def __init__(self, environment):
        """

        :param environment:
        """
        self.environment = environment

    def maximize_reward_from_price(self, category):
        """

        :param category:
        :return:
        """
        values = np.array([])
        for idx, price in enumerate(self.environment.arms_values[category]):
            values = np.append(values, self.environment.probabilities[category][idx] * (price - self.environment.other_costs))

        best_price_idx = np.random.choice(np.where(values == values.max())[0])

        return best_price_idx, self.environment.arms_values[category][best_price_idx], values[best_price_idx]

    def maximize_reward_from_bid(self, category, conversion_times_margin):
        """

        :param category:
        :param conversion_times_margin:
        :return:
        """
        values = np.array([])
        for bid in self.environment.bids:
            n_clicks = fun(bid, *self.environment.bids_to_clicks[category])
            cum_daily_costs = fun(bid, *self.environment.bids_to_cum_costs[category])
            values = np.append(values, n_clicks * conversion_times_margin - cum_daily_costs)

        best_bid_idx = np.random.choice(np.where(values == values.max())[0])

        return best_bid_idx, self.environment.bids[best_bid_idx], values[best_bid_idx]

    def maximize_reward(self, category):
        """

        :param category:
        :return:
        """
        best_price_idx, best_price, conversion_times_margin = self.maximize_reward_from_price(category)
        best_bid_idx, best_bid, reward = self.maximize_reward_from_bid(category, conversion_times_margin)

        return best_price_idx, best_price, best_bid_idx, best_bid, reward


category = 'C1'
n_arms = 5
arms_values = {'C1': np.array([500, 550, 600, 650, 700]),
               'C2': np.array([500, 550, 600, 650, 700]),
               'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.01, 0.01, 0.05, 0.02, 0.01]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([1, 1, 0.5]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([10, 0.5, 0.5]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 400
env = Environment(n_arms, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
clairvoyant = Clairvoyant(env)
#clairvoyant.maximize_reward_from_price(category)
#print(clairvoyant.maximize_reward_from_price(category))
#print(str(clairvoyant.maximize_reward(category)))
#env.plot_whole_advertising_model()

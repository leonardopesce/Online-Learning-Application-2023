from Environment import *


class Clairvoyant:
    """
    Learner that knows the model of the environment and so can play the best values of price to solve the pricing
    problem and the best values of the bid to solve the advertising problem

    environment: Environment where has to be solved the problem
    """

    def __init__(self, environment):
        """
        Initialize the learner given the environment to solve

        :param Environment environment: Environment where has to be solved the problem
        """

        self.environment = environment

    def maximize_reward_from_price(self, category):
        """
        Find the price, considering a single category, that maximizes the part of the reward depending on the pricing
        problem independently with respect to the advertising quantities, thus it finds the price that maximizes the
        conversion probability multiplied by the margin

        :param str category: Category considered in the maximization of the reward
        :return: Index of the best price in the list of the prices, value of the best price, value of the product
        conversion probability times the margin using the best price
        :rtype: tuple
        """
        values = np.array([])
        for idx, price in enumerate(self.environment.arms_values[category]):
            values = np.append(values, self.environment.probabilities[category][idx] * (price - self.environment.other_costs))

        best_price_idx = np.random.choice(np.where(values == values.max())[0])

        return best_price_idx, self.environment.arms_values[category][best_price_idx], values[best_price_idx]

    def maximize_reward_from_bid(self, category, conversion_times_margin):
        """
        Find the bid, considering a single category, that maximizes the reward, defined as the number of daily clicks
        multiplied by the conversion probability multiplied by the margin minus the cumulative daily costs due to the
        advertising, given the product between the conversion probability and the margin

        :param str category: Category considered in the maximization of the reward
        :param float conversion_times_margin: Conversion probability multiplied by the margin to use in the computation
        :return: Index of the best bid in the list of the bids, value of the best bid, value of the reward using the
        best bid and the given product between conversion probability and the margin
        :rtype: tuple
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
        Find the price and the bid, considering a single category, that maximize the reward, defined as the number of
        daily clicks multiplied by the conversion probability multiplied by the margin minus the cumulative daily costs
        due to the advertising

        :param str category: Category considered in the maximization of the reward
        :return: Index of the best price in the list of the prices, value of the best price, index of the best bid in
        the list of the bids, value of the best bid, value of the reward using the best price and the best bid when
        computing it
        :rtype: tuple
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

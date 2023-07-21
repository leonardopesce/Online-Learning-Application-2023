import numpy as np
import matplotlib.pyplot as plt


def fun(x, scale, slope, starting_value):
    """
    It returns a logarithmic function used to model the advertising setting

    :param float x: Value of independent variable, in this setting, it is the bid
    :param float scale: Parameter that affects the scale of the function
    :param float slope: Parameter that affects the slope of the function
    :param float starting_value: Value before which the function is zero
    :return: Value assumed by the logarithmic function
    :rtype: float
    """

    return scale * np.log(slope * (x + 1 / slope - starting_value))


# TODO che facciamo con questa, provarla?
def fun1(x):
    return 1000 * (1.0 - np.exp(-2 * x))
    #return 100 * (1.0 - np.exp(-4*x+3*x**3))


class Environment:
    """
    The Environment class defines the advertising and pricing environment using for each class the models of:
    - the average dependence between the number of clicks and the bid;
    - the average cumulative daily click cost for the bid;
    - the conversion rate for 5 different prices.
    The Environment class allows the agents to interact with it using its functions

    :param int n_prices: Number of arms
    :param dict arms_values: Dictionary that maps each class of users to the values(price of the product) associated to the arms
    :param dict probabilities: Dictionary that maps each class to the bernoulli probabilities associated to the arms
    bids: array of 100 possible bid values
    :param dict bids_to_clicks: Dictionary that maps each class to the parameters to build the function that models the number of clicks given the bid
    bids_to_clicks_variance: Variance of the gaussian noise associated to the function that models the number of clicks given the bid
    :param dict bids_to_cum_costs: Dictionary that maps each class to the parameters to build the function that models the cumulative daily click cost given the bid
    bids_to_cum_costs_variance: Variance of the gaussian noise associated to the function that models the cumulative daily click cost given the bid
    :param float other_costs: Cost of the product
    """

    def __init__(self, n_prices, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs):
        self.n_prices = n_prices
        self.arms_values = arms_values
        self.probabilities = probabilities
        self.bids = np.linspace(0.5, 15, 100)
        self.bids_to_clicks = bids_to_clicks
        self.bids_to_clicks_variance = 0.2
        self.bids_to_cum_costs = bids_to_cum_costs
        self.bids_to_cum_costs_variance = 0.2
        self.other_costs = other_costs

    def round_pricing(self, pulled_arm, category):
        """
        Simulates a round in a pricing scenario, returning the realization of the pulled arm for the given class of users

        :param int pulled_arm: Arm pulled in the current time step
        :param str category: Class of the user
        :return: Realization of the pulled arm, either 0(not buy) or 1(buy)
        :rtype: int
        """

        realization = np.random.binomial(1, self.probabilities[category][pulled_arm])

        return realization

    def round_advertising(self, bid_idx, category):
        """
        Simulates a round in an advertising scenario, returning the number of clicks and the cumulative daily click cost
        given the bid and the class of users

        :param int bid_idx: Index of the bid used in the current round
        :param str category: Class of the user
        :return: Number of clicks, cumulative daily click cost
        :rtype: tuple
        """

        clicks_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_clicks[category]) + np.random.randn() * np.sqrt(self.bids_to_clicks_variance))
        cost_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_cum_costs[category]) + np.random.randn() * np.sqrt(self.bids_to_cum_costs_variance))

        return clicks_given_bid, cost_given_bid

    def round(self, pulled_arm, bid_idx, category):
        """
        Simulates a round in a pricing-advertising scenario, returning the realization of the pulled arm,
        number of clicks and cumulative daily click cost given the pulled arm, the bid and the class of users

        :param int pulled_arm: Arm pulled in the current time step
        :param int bid_idx: Index of the bid used in the current round
        :param str category: Class of the user
        :return: Realization of the pulled arm, number of clicks, cumulative daily click cost
        :rtype: tuple
        """

        return self.round_pricing(pulled_arm, category), self.round_advertising(bid_idx, category)

    def get_n_clicks(self, category, bid_idx):
        """
        Return the number of clicks given the bid and class of the user

        :param str category: Class of the user
        :param bid_idx: Index of the bid
        :return: Number of clicks
        :rtype: float
        """

        return fun(self.bids[bid_idx], *self.bids_to_clicks[category])

    def get_cum_daily_costs(self, category, bid_idx):
        """
        Return the cumulative daily costs due to advertising given the bid and class of the user

        :param str category: Class of the user
        :param bid_idx: Index of the bid
        :return: Cumulative daily costs
        :rtype: float
        """

        return fun(self.bids[bid_idx], *self.bids_to_cum_costs[category])

    def get_reward(self, category, price_idx, n_clicks, cum_daily_costs, conversion_prob=None):
        """
        Compute the reward defined as the number of daily clicks multiplied by the conversion probability multiplied by
        the margin minus the cumulative daily costs due to the advertising

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_prob: Conversion probability
        :param float n_clicks: Number of daily clicks
        :param float cum_daily_costs: Cumulative daily cost due to the advertising
        :return: Reward
        :rtype: float
        """
        conversion_prob = self.probabilities[category][price_idx] if conversion_prob is None else conversion_prob

        return n_clicks * conversion_prob * (self.arms_values[category][price_idx] - self.other_costs) - cum_daily_costs

    def get_reward_from_price(self, category, price_idx, conversion_prob, bid_idx):
        """
        Compute the reward defined as the number of daily clicks multiplied by the conversion probability multiplied by
        the margin minus the cumulative daily costs due to the advertising. The advertising scenario si assumed to be known

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_prob: Conversion probability
        :param bid_idx: Index of the bid
        :return: Reward
        :rtype: float
        """

        n_clicks = fun(self.bids[bid_idx], *self.bids_to_clicks[category])
        cum_daily_costs = fun(self.bids[bid_idx], *self.bids_to_cum_costs[category])
        return n_clicks * conversion_prob * (self.arms_values[category][price_idx] - self.other_costs) - cum_daily_costs

    def get_conversion_times_margin(self, category, price_idx, conversion_probability=None):
        """
        Compute the product between the conversion probability and the margin

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_probability: If not given the value from the model is used, otherwise you can pass the realization of the arm
        :return: Product between the conversion probability and the margin
        :rtype: float
        """
        if conversion_probability is None:
            return self.probabilities[category][price_idx] * (self.arms_values[category][price_idx] - self.other_costs)
        return conversion_probability * (self.arms_values[category][price_idx] - self.other_costs)

    def plot_pricing_model(self, category, color='r', axes=None, show=True):
        """
        Plot the bernoulli probabilities associated to the arms of the given class of users

        :param str category: Class of the user
        :param str color: Color of the plot
        :param plt.Axes axes: Axes of the plot
        :param bool show: Boolean to show the plot
        :return: Axes of the plot
        :rtype: plt.Axes
        """

        if axes is None:
            _, axes = plt.subplots(1, 1)

        axes.plot(self.arms_values[category], self.probabilities[category], color=color, label=category)
        axes.set_title('Conversion rate of the arms')
        axes.set_xlabel('Arm value')
        axes.set_ylabel('Conversion rate')
        axes.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return axes

    def plot_whole_pricing_model(self, mapping={'C1': 'r', 'C2': 'b', 'C3': 'g'}):
        """
        Plot the bernoulli probabilities associated to the arms of all classes of users

        :param dict mapping: Mapping between classes of users and colors to use for the plot
        :return: Axes of the plot
        :rtype: plt.Axes
        """

        _, axes = plt.subplots(1, 1)
        for category in self.bids_to_clicks.keys():
            axes = self.plot_pricing_model(category, color=mapping[category], axes=axes, show=False)

        plt.tight_layout()
        plt.show()

        return axes

    def plot_advertising_model(self, category, xlim=15, color='r', axes=None, show=True):
        """
        Plot the function that models the number of clicks given the bid and the function that models the cumulative
        daily click cost given the bid, for the given class of users

        :param str category: Class of the user
        :param float xlim: Max value for the x-axis
        :param str color: Color of the plot
        :param plt.Axes axes: Axes of the plot
        :param bool show: Boolean to show the plot
        :return: Axes of the plot
        :rtype: plt.Axes
        """

        if axes is None:
            _, axes = plt.subplots(1, 2)

        x = np.linspace(0, self.bids_to_clicks[category][2], 100)
        y = np.zeros((100,))
        axes[0].plot(x, y, color=color)
        x = np.linspace(self.bids_to_clicks[category][2], xlim, 100)
        y = fun(x, *self.bids_to_clicks[category])
        axes[0].plot(x, y, color=color, label=category)
        axes[0].set_title('Clicks given the bid')
        axes[0].set_xlabel('Value of the bid')
        axes[0].set_ylabel('Number of clicks')
        axes[0].legend()

        x = np.linspace(0, self.bids_to_cum_costs[category][2], 100)
        y = np.zeros((100,))
        axes[1].plot(x, y, color=color)
        x = np.linspace(self.bids_to_cum_costs[category][2], xlim, 100)
        y = fun(x, *self.bids_to_cum_costs[category])
        axes[1].plot(x, y, color=color, label=category)
        axes[1].set_title('Cost of the clicks given the bid')
        axes[1].set_xlabel('Value of the bid')
        axes[1].set_ylabel('Cumulative cost of the clicks')
        axes[1].legend()

        if show:
            plt.tight_layout()
            plt.show()

        return axes

    def plot_whole_advertising_model(self, xlim=15, mapping={'C1': 'r', 'C2': 'b', 'C3': 'g'}):
        """
        Plot the function that models the number of clicks given the bid and the function that models the cumulative
        daily click cost given the bid, for all the classes of users

        :param xlim: Max value for the x-axis
        :param dict mapping: Mapping between classes of users and colors to use for the plot
        :return: Axes of the plot
        :rtype: plt.Axes
        """

        _, axes = plt.subplots(1, 2)
        for category in self.bids_to_clicks.keys():
            axes = self.plot_advertising_model(category, xlim=xlim, color=mapping[category], axes=axes, show=False)

        plt.tight_layout()
        plt.show()

        return axes


def test():
    # TESTING
    n_prices = 5
    arms_values = {'C1': np.array([500, 550, 600, 650, 700]),
                   'C2': np.array([500, 550, 600, 650, 700]),
                   'C3': np.array([500, 550, 600, 650, 700])}
    probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                     'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                     'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
    bids_to_clicks = {'C1': np.array([3, 1, 0.5]),
                      'C2': np.array([2, 2, 0.5]),
                      'C3': np.array([3, 3, 0.5])}
    bids_to_cum_costs = {'C1': np.array([10, 0.5, 0.5]),
                         'C2': np.array([2, 2, 0.5]),
                         'C3': np.array([3, 3, 0.5])}
    other_costs = 200
    env = Environment(n_prices, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
    env.plot_advertising_model('C1', color='r', axes=None)
    env.plot_whole_advertising_model()
    env.plot_whole_pricing_model()


# test()

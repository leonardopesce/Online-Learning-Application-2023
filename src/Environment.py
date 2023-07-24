import numpy as np
import matplotlib.pyplot as plt


def fun(x, scale, slope):
    """
    It returns an exponential function used to model the advertising setting

    :param float x: Value of independent variable, in this setting, it is the bid
    :param float scale: Parameter that affects the scale of the function
    :param float slope: Parameter that affects the slope of the function
    :return: Value assumed by the exponential function
    :rtype: float
    """

    #return scale * np.log(slope * (x + 1 / slope - starting_value))
    return scale * (1.0 - np.exp(-slope * x))


class Environment:
    """
    The Environment class defines the advertising and pricing environment usin, for each class, the models of:
    - the average dependence between the number of clicks and the bid;
    - the average cumulative daily click cost for the bid;
    - the conversion rate for 5 different prices.
    The Environment class allows the agents to interact with it using its functions

    Attributes:
        n_prices: Number of prices
        prices: Dictionary that maps each class of users to the values(price of the product) associated to the arms
        probabilities: Dictionary that maps each class to the bernoulli probabilities associated to the arms
        bids: Array of 100 possible bid values
        bids_to_clicks: Dictionary that maps each class to the parameters to build the function that models the number of clicks given the bid
        bids_to_clicks_variance: Variance of the gaussian noise associated to the function that models the number of clicks given the bid
        bids_to_cum_costs: Dictionary that maps each class to the parameters to build the function that models the cumulative daily click cost given the bid
        bids_to_cum_costs_variance: Variance of the gaussian noise associated to the function that models the cumulative daily click cost given the bid
        other_costs: Cost of the product
    """

    def __init__(self, n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs):
        """
        Initializes the Environment class

        :param int n_prices: Number of prices
        :param dict prices: Dictionary that maps each class of users to the values(price of the product) associated to the arms
        :param dict probabilities: Dictionary that maps each class to the bernoulli probabilities associated to the arms
        :param dict bids_to_clicks: Dictionary that maps each class to the parameters to build the function that models the number of clicks given the bid
        :param dict bids_to_cum_costs: Dictionary that maps each class to the parameters to build the function that models the cumulative daily click cost given the bid
        :param float other_costs: Cost of the product
        """

        self.n_prices = n_prices
        self.prices = prices
        self.probabilities = probabilities
        self.bids = np.linspace(0.5, 15, 100)
        self.bids_to_clicks = bids_to_clicks
        self.bids_to_clicks_variance = 0.2
        self.bids_to_cum_costs = bids_to_cum_costs
        self.bids_to_cum_costs_variance = 0.2
        self.other_costs = other_costs

    def round_pricing(self, category, price_idx, n_clicks=1):
        """
        Simulates a round in a pricing scenario, returning the realization of the pulled price for the given class of
        users

        :param str category: Class of the user
        :param int price_idx: Arm pulled in the current time step
        :param int n_clicks: Number of clicks in the current time step, so number of observation to draw from the Bernoulli

        :return: Realization of the pulled arm, either 0(not buy) or 1(buy)
        :rtype: numpy.ndarray
        """

        realizations = np.array([np.random.binomial(1, self.probabilities[category][price_idx]) for _ in range(0, int(n_clicks))])
        if len(realizations) == 0:
            realizations = np.array([0])

        return realizations

    def round_advertising(self, category, bid_idx):
        """
        Simulates a round in an advertising scenario, returning the number of clicks and the cumulative daily click cost
        given the bid and the class of users

        :param str category: Class of the user
        :param int bid_idx: Index of the bid used in the current round

        :return: Number of clicks, cumulative daily click cost
        :rtype: tuple
        """

        clicks_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_clicks[category]) + np.random.randn() * np.sqrt(self.bids_to_clicks_variance))
        cost_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_cum_costs[category]) + np.random.randn() * np.sqrt(self.bids_to_cum_costs_variance))

        return clicks_given_bid, cost_given_bid

    def round(self, category, price_idx, bid_idx):
        """
        Simulates a round in a pricing-advertising scenario, returning the realization of the chosen price, number of
        clicks and cumulative daily click cost given the price, the bid and the class of users

        :param str category: Class of the user
        :param int price_idx: Arm pulled in the current time step
        :param int bid_idx: Index of the bid used in the current round

        :return: Realization of the pulled price, number of clicks, cumulative daily click cost
        :rtype: tuple
        """

        clicks_given_bid, cost_given_bid = self.round_advertising(category, bid_idx)
        bernoulli_realizations = self.round_pricing(category, price_idx, n_clicks=int(np.floor(clicks_given_bid)))

        return bernoulli_realizations, clicks_given_bid, cost_given_bid

    def get_n_clicks(self, category, bid_idx):
        """
        Returns the number of clicks given the bid and class of the user

        :param str category: Class of the user
        :param int bid_idx: Index of the bid

        :return: Number of clicks
        :rtype: float
        """

        return fun(self.bids[bid_idx], *self.bids_to_clicks[category])

    def get_cum_daily_costs(self, category, bid_idx):
        """
        Returns the cumulative daily costs due to advertising given the bid and class of the user

        :param str category: Class of the user
        :param int bid_idx: Index of the bid

        :return: Cumulative daily costs
        :rtype: float
        """

        return fun(self.bids[bid_idx], *self.bids_to_cum_costs[category])

    def get_reward(self, category, price_idx, conversion_prob, n_clicks, cum_daily_costs):
        """
        Computes the reward defined as the number of daily clicks multiplied by the conversion probability multiplied by
        the margin minus the cumulative daily costs due to the advertising

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_prob: Conversion probability
        :param float n_clicks: Number of daily clicks
        :param float cum_daily_costs: Cumulative daily cost due to the advertising

        :return: Reward
        :rtype: float
        """

        return n_clicks * conversion_prob * (self.prices[category][price_idx] - self.other_costs) - cum_daily_costs

    def get_reward_from_price(self, category, price_idx, conversion_prob, bid_idx):
        """
        Computes the reward defined as the number of daily clicks multiplied by the conversion probability multiplied by
        the margin minus the cumulative daily costs due to the advertising. The advertising scenario si assumed to be
        known

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_prob: Conversion probability
        :param bid_idx: Index of the bid

        :return: Reward
        :rtype: float
        """

        n_clicks = fun(self.bids[bid_idx], *self.bids_to_clicks[category])
        cum_daily_costs = fun(self.bids[bid_idx], *self.bids_to_cum_costs[category])
        return n_clicks * conversion_prob * (self.prices[category][price_idx] - self.other_costs) - cum_daily_costs

    def get_conversion_times_margin(self, category, price_idx, conversion_probability=None):
        """
        Computes the product between the conversion probability and the margin

        :param str category: Class of the user
        :param int price_idx: Index of the price
        :param float conversion_probability: If not given the value from the model is used, otherwise you can pass the
        realization of the price that is the conversion probability

        :return: Product between the conversion probability and the margin
        :rtype: float
        """

        if conversion_probability is None:
            return self.probabilities[category][price_idx] * (self.prices[category][price_idx] - self.other_costs)

        return conversion_probability * (self.prices[category][price_idx] - self.other_costs)

    def plot_pricing_model(self, category, color='r', axes=None, show=True):
        """
        Plots the Bernoulli probabilities associated to the arms of the given class of users

        :param str category: Class of the user
        :param str color: Color of the plot
        :param plt.Axes axes: Axes of the plot, by default it is None and in this case new axes are generated
        :param bool show: Boolean to show the plot

        :return: Axes of the plot
        :rtype: plt.Axes
        """

        if axes is None:
            _, axes = plt.subplots(1, 1)

        axes.plot(self.prices[category], self.probabilities[category], color=color, label=category)
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
        Plots the Bernoulli probabilities associated to the arms of all classes of users

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
        Plots the function that models the number of clicks given the bid and the function that models the cumulative
        daily click cost given the bid, for the given class of users

        :param str category: Class of the user
        :param float xlim: Max value for the x-axis
        :param str color: Color of the plot
        :param plt.Axes axes: Axes of the plot, by default it is None and in this case new axes are generated
        :param bool show: Boolean to show the plot

        :return: Axes of the plot
        :rtype: plt.Axes
        """

        if axes is None:
            _, axes = plt.subplots(1, 2)

        #x = np.linspace(0, self.bids_to_clicks[category][2], 100)
        #y = np.zeros((100,))
        #axes[0].plot(x, y, color=color)
        x = np.linspace(0, xlim, 100)
        y = fun(x, *self.bids_to_clicks[category])
        axes[0].plot(x, y, color=color, label=category)
        axes[0].set_title('Clicks given the bid')
        axes[0].set_xlabel('Value of the bid')
        axes[0].set_ylabel('Number of clicks')
        axes[0].legend()

        #x = np.linspace(0, self.bids_to_cum_costs[category][2], 100)
        #y = np.zeros((100,))
        #axes[1].plot(x, y, color=color)
        x = np.linspace(0, xlim, 100)
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
        Plots the function that models the number of clicks given the bid and the function that models the cumulative
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
    prices = {'C1': np.array([500, 550, 600, 650, 700]),
              'C2': np.array([500, 550, 600, 650, 700]),
              'C3': np.array([500, 550, 600, 650, 700])}
    probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                     'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                     'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
    bids_to_clicks = {'C1': np.array([100, 2]),
                      'C2': np.array([90, 2]),
                      'C3': np.array([80, 3])}
    bids_to_cum_costs = {'C1': np.array([20, 0.5]),
                         'C2': np.array([18, 0.4]),
                         'C3': np.array([16, 0.45])}
    other_costs = 200

    env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
    env.plot_advertising_model('C1', color='r', axes=None)
    env.plot_whole_advertising_model()
    env.plot_whole_pricing_model()


#test()

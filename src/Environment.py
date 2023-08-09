import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    The Environment class defines the advertising and pricing environment using, for each class, the models of:
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

    # TODO probabilmente sarebbe meglio rendere environment uno per classe e creare un environment con multiple contexts
    def round_all_categories_merged(self, price_idx, bid_idx):
        categories = self.probabilities.keys()
        bernoulli_realizations_all_categories = np.array([])
        clicks_given_bid_all_categories = 0
        cost_given_bid_all_categories = 0
        for category in categories:
            bernoulli_realizations, clicks_given_bid, cost_given_bid = self.round(category, price_idx, bid_idx)
            bernoulli_realizations_all_categories = np.append(bernoulli_realizations_all_categories, bernoulli_realizations)
            clicks_given_bid_all_categories += clicks_given_bid
            cost_given_bid_all_categories += cost_given_bid

        return bernoulli_realizations_all_categories, clicks_given_bid_all_categories, cost_given_bid_all_categories

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
        #conversion_prob = self.probabilities[category][price_idx] if conversion_prob is None else conversion_prob

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
            _, axes = plt.subplots(1, 3)

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

        x = np.linspace(0, xlim, 100)
        y = fun(x, *self.bids_to_clicks[category]) * x - fun(x, *self.bids_to_cum_costs[category])
        if any(y < 0):
            print(f'For the category {category} some bids do not respect the consistency check (Number of clicks times bid > cumulative daily cost)')
        axes[2].plot(x, y, color=color, label=category)
        axes[2].set_title('Number of clicks times bid minus daily cost (Consistency check)')
        axes[2].set_xlabel('Value of the bid')
        axes[2].set_ylabel('Number of clicks times bid minus daily cost')
        axes[2].legend()

        if show:
            plt.tight_layout()
            plt.show()

        return axes

    def plot_whole_advertising_model(self, xlim=15, mapping={'C1': 'r', 'C2': 'b', 'C3': 'g'}):
        """
        Plots the function that models the number of clicks given the bid and the function that models the cumulative
        daily click cost given the bid, for all the classes of users

        :param float xlim: Max value for the x-axis
        :param dict mapping: Mapping between classes of users and colors to use for the plot

        :return: Axes of the plot
        :rtype: plt.Axes
        """

        _, axes = plt.subplots(1, 3, figsize=(18,6))
        for category in self.bids_to_clicks.keys():
            axes = self.plot_advertising_model(category, xlim=xlim, color=mapping[category], axes=axes, show=False)

        plt.tight_layout()
        plt.show()

        return axes

    def plot_rewards_given_price_idx(self, price_idx, xlim=15, mapping={'C1': 'r', 'C2': 'b', 'C3': 'g'}):
        """
        Plots the reward with respect to the bid given the price index to use in the computation

        :param int price_idx: Index of the price to use in the computation
        :param float xlim: Max value for the x-axis
        :param dict mapping: Mapping between classes of users and colors to use for the plot

        :return: Axes of the plot
        :rtype: plt.Axes
        """

        _, axes = plt.subplots(1, 1)

        x = np.linspace(0, xlim, 100)
        for category in self.bids_to_clicks.keys():
            y = fun(x, *self.bids_to_clicks[category]) * self.probabilities[category][price_idx] * (self.prices[category][price_idx] - self.other_costs) - fun(x, *self.bids_to_cum_costs[category])
            axes.plot(x, y, color=mapping[category], label=category)
            axes.axvline(x[np.argmax(y)], color=mapping[category], label=f'Max of {category}')
            axes.set_title(f'Reward given the bid, price = {self.prices[category][price_idx]}')
            axes.set_xlabel('Value of the bid')
            axes.set_ylabel('Reward')
            axes.legend()

        plt.tight_layout()
        plt.show()

        return axes

    def plot_rewards(self, categories=('C1', 'C2', 'C3'), plot_aggregate_model=False):
        """
        Plots the reward with respect to the bid given the price index to use in the computation

        :param tuple categories: Categories for which the plot has to be done
        :param bool plot_aggregate_model: If True the reward for the aggregate model of all the classes is plotted

        :return: Axes of the plot
        :rtype: plt.Axes
        """

        aggregate_model = np.zeros((len(self.bids), self.n_prices))

        for category in self.bids_to_clicks:
            x = self.bids
            y = self.prices[category]
            xx, yy = np.meshgrid(x, y)

            n_clicks = fun(x, *self.bids_to_clicks[category])
            cum_costs = fun(x, *self.bids_to_cum_costs[category])
            conv_prob = self.probabilities[category]

            z = n_clicks[:, None] * conv_prob[None, :] * (self.prices[category] - self.other_costs) - cum_costs[:, None]

            if plot_aggregate_model:
                aggregate_model += z

            if category in categories:
                _, axes = plt.subplots(subplot_kw={'projection': '3d'})
                axes.plot_surface(xx, yy, z.T, cmap='viridis')

                argmax_x = np.argmax(z) // self.n_prices
                argmax_y = np.argmax(z) % self.n_prices
                argmax_z = np.max(z)

                print(f'For the category {category} the maximum is in (bid, price) = ({self.bids[argmax_x]}, {self.prices[category][argmax_y]}), with a value of {argmax_z}')
                axes.scatter(self.bids[argmax_x], self.prices[category][argmax_y], argmax_z, color='black', s=20, label=f'Max of {category}')
                axes.set_title(f'Reward given price and bid for the user category {category}')
                axes.set_xlabel('Value of the bid')
                axes.set_ylabel('Value of the price')
                axes.set_zlabel('Reward')

                plt.tight_layout()
                plt.show()

        if plot_aggregate_model:
            _, axes = plt.subplots(subplot_kw={'projection': '3d'})

            x = self.bids
            y = self.prices[category]
            xx, yy = np.meshgrid(x, y)

            axes.plot_surface(xx, yy, aggregate_model.T, cmap='viridis')

            argmax_x = np.argmax(aggregate_model) // self.n_prices
            argmax_y = np.argmax(aggregate_model) % self.n_prices
            argmax_z = np.max(aggregate_model)

            print(f'For the aggregate model the maximum is in (bid, price) = ({self.bids[argmax_x]}, {self.prices["C1"][argmax_y]}), with a value of {argmax_z}')
            axes.scatter(self.bids[argmax_x], self.prices[category][argmax_y], argmax_z, color='black', s=20, label=f'Max of {category}')
            axes.set_title(f'Reward given price and bid using the aggregate model')
            axes.set_xlabel('Value of the bid')
            axes.set_ylabel('Value of the price')
            axes.set_zlabel('Reward')

            plt.tight_layout()
            plt.show()

def test():
    # TESTING

    n_prices = 5
    prices = {'C1': np.array([500, 550, 600, 650, 700]),
              'C2': np.array([500, 550, 600, 650, 700]),
              'C3': np.array([500, 550, 600, 650, 700])}
    probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                     'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                     'C3': np.array([0.1, 0.2, 0.25, 0.05, 0.05])}
    bids_to_clicks = {'C1': np.array([100, 2]),
                      'C2': np.array([90, 2]),
                      'C3': np.array([80, 3])}
    bids_to_cum_costs = {'C1': np.array([400, 0.035]),
                         'C2': np.array([200, 0.07]),
                         'C3': np.array([300, 0.04])}
    other_costs = 400

    env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
    #env.plot_advertising_model('C1', color='r', axes=None)
    env.plot_whole_advertising_model()
    #env.plot_whole_pricing_model()
    env.plot_rewards_given_price_idx(2)
    env.plot_rewards(categories=['C1'], plot_aggregate_model=True)


# test()

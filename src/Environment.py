import numpy as np
import matplotlib.pyplot as plt


def fun(x, scale, slope, starting_value):
    """
    It returns a logarithmic function used to model the advertising setting.

    :param x: value of independent variable, in this setting, it is the bid.
    :param scale: parameter that affects the scale of the function.
    :param slope: parameter that affects the slope of the function.
    :param starting_value: value before which the functions are zero.
    :return: value assumed by the logarithmic function.
    """

    return scale * np.log(slope * (x + 1 / slope - starting_value))


class Environment:
    """
    The Environment class defines the advertising and pricing environment using for each class the models of:
    - the average dependence between the number of clicks and the bid;
    - the average cumulative daily click cost for the bid;
    - the conversion rate for 5 different prices.
    The Environment class allows the agents to interact with it using its functions.

    # TO DO specify what are the parameters
    """

    def __init__(self):#, bids_to_clicks, bids_to_cum_costs, arms_values, probabilities): # TO DO rendiamo parametrica la definizione degli attributi
        self.n_arms = 5
        self.arms_values = {'C1': np.array([500, 550, 600, 650, 700]),
                            'C2': np.array([500, 550, 600, 650, 700]),
                            'C3': np.array([500, 550, 600, 650, 700])}
        self.probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                              'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                              'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
        self.bids = np.linspace(0.5, 20, 100)
        self.bids_to_clicks = {'C1': np.array([1, 1, 0.5]),
                               'C2': np.array([2, 2, 0.5]),
                               'C3': np.array([3, 3, 0.5])}
        self.bids_to_clicks_variance = 0.2
        self.bids_to_cum_costs = {'C1': np.array([100, 0.5, 0.5]),
                                  'C2': np.array([2, 2, 0.5]),
                                  'C3': np.array([3, 3, 0.5])}
        self.bids_to_cum_costs_variance = 0.2
        self.other_costs = 200

    def round_pricing(self, pulled_arm, category):
        """

        :param pulled_arm:
        :param category:
        :return:
        """

        reward = np.random.binomial(1, self.probabilities[category][pulled_arm])

        return reward

    def round_advertising(self, bid_idx, category):
        """
        Simulates the advertising model returning the number of clicks given the bid and the cumulative daily click cost
        given the bid.
        :param bid_idx: index of the bid used in the current round
        :param category: category for which it is needed to
        :return:
        """
        # Forse al posto di bid dovremmo mettere degli indici e fissare dei valori di bid giocabili? DA CAMBIARE

        clicks_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_clicks[category]) + np.random.randn() * np.sqrt(self.bids_to_clicks_variance))
        cost_given_bid = max(0, fun(self.bids[bid_idx], *self.bids_to_cum_costs[category]) + np.random.randn() * np.sqrt(self.bids_to_cum_costs_variance))

        return clicks_given_bid, cost_given_bid

    def round(self, pulled_arm, bid, category):
        """

        :param pulled_arm:
        :param bid:
        :param category:
        :return:
        """

        return self.round_pricing(pulled_arm, category), self.round_advertising(bid, category)

    def plot_pricing_model(self, category, color='r', axes=None, show=True):
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

        :param mapping:
        :return:
        """

        _, axes = plt.subplots(1, 1)
        for category in self.bids_to_clicks.keys():
            axes = self.plot_pricing_model(category, color=mapping[category], axes=axes, show=False)

        plt.tight_layout()
        plt.show()

        return axes

    def plot_advertising_model(self, category, xlim=15, color='r', axes=None, show=True):
        """

        :param category:
        :param xlim:
        :param color:
        :param axes:
        :param show:
        :return:
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

        :param xlim:
        :param mapping:
        :return:
        """

        _, axes = plt.subplots(1, 2)
        for category in self.bids_to_clicks.keys():
            axes = self.plot_advertising_model(category, xlim=xlim, color=mapping[category], axes=axes, show=False)

        plt.tight_layout()
        plt.show()

        return axes

    def reward(self, category, price_idx, conversion_prob, n_clicks, cum_daily_costs):
        """
        the reward is defined as the number of daily clicks multiplied by the conversion probability multiplied by the
        margin minus the cumulative daily costs due to the advertising.

        :return:
        """

        return n_clicks * conversion_prob * (self.arms_values[category][price_idx] - self.other_costs) - cum_daily_costs


def test():
    # TESTING
    env = Environment()
    env.plot_advertising_model('C1', color='r', axes=None)
    env.plot_whole_advertising_model()
    env.plot_whole_pricing_model()

test()
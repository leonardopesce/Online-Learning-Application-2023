from Environment import *


class MultiContextEnvironment(Environment):
    """
    The MultiContextEnvironment class defines the advertising and pricing environment using, for each class, the models of:
    - the average dependence between the number of clicks and the bid;
    - the average cumulative daily click cost for the bid;
    - the conversion rate for 5 different prices.

    The MultiContextEnvironment class allows the agents to interact with it using its functions.
    From outside the structure of the context is not visible and the class allows to have an interaction with an
    environment with multiple contexts.

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

    def __init__(self, n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, categories, feature_names, feature_values, feature_values_to_categories, probability_feature_values_in_categories):
        """
        The
        """

        super().__init__(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)

        self.categories = categories
        self.feature_name = feature_names
        self.feature_values = feature_values
        self.feature_values_to_categories = feature_values_to_categories
        self.probability_feature_values_in_categories = probability_feature_values_in_categories

    def get_reward(self, context, price_idx, conversion_prob, n_clicks, cum_daily_costs):
        tmp = conversion_prob * (self.prices['C1'][price_idx] - self.other_costs) * n_clicks - cum_daily_costs
        return tmp

    def round(self, price_idx, bid_idx, user_features_set):
        # price_idx, bid_idx = learner.pull_arms()
        # bernoulli_realizations, clicks_given_bid, cost_given_bid = environment.round(price_idx, bid_idx, learner.get_context_features() (che potrebbe tornare set((0,0), (0,1))))
        """
        :param set tuple user_features_set: Tuple containing the features of the
        """

        # TODO ATTENZIONE DA VEDERE: dobbiamo analizzare bene questo fatto: data la nostra implementazione, se giocassimo
        # TODO un round per ogni set di features mappandole nella rispettiva categoria e poi estraendo ( da round in Environment) una parte delle osservazioni
        # TODO basandosi su categories_to_feature_values potremmo dover buttare via parte di esse, è logicamente/computazionalmente sensato farlo?
        # TODO Abbiamo necessità di fare round in MultiContextEnvironment basandoci su set di features perchè il learner necessiteràdi giocare queste.

        # TODO To check: ha senso tornare il numero di click e il cost aggregato su tutte le tuple di user_features o no?
        # TODO Secondo noi ha più senso separare anche nell'advertiseing sulla base delle features (come stiamo facendo in questo metodo)

        features_list = []
        bernoulli_realizations_list = []
        clicks_given_bid_list = []
        cost_given_bid_list = []

        for user_features in user_features_set:
            category = self.feature_values_to_categories[user_features]
            probability_of_features = self.probability_feature_values_in_categories[category][user_features]

            bernoulli_realizations, clicks_given_bid, cost_given_bid = super().round(category, price_idx, bid_idx)

            features_list.append(user_features)
            clicks_given_features = int(np.ceil(probability_of_features * clicks_given_bid))
            clicks_given_bid_list.append(clicks_given_features)
            cost_given_features = probability_of_features * cost_given_bid
            cost_given_bid_list.append(cost_given_features)
            bernoulli_realizations_list.append(np.random.choice(bernoulli_realizations, clicks_given_features))

        return features_list, bernoulli_realizations_list, clicks_given_bid_list, cost_given_bid_list


if __name__ == '__main__':
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

    categories = ['C1', 'C2', 'C3']
    # C1: young 0, man 1; C2: old 1, man 1; C3: [0,1], woman 0
    feature_names = ['age', 'sex']
    feature_values = {'age': [0, 1], 'sex': [0, 1]}
    # age: 0 -> young, 1 -> old; sex: 0 -> woman, 1 -> man
    feature_values_to_categories = {(0, 0): 'C3', (0, 1): 'C1', (1, 0): 'C3', (1, 1): 'C2'}
    probability_feature_values_in_categories = {'C1': {(0, 1): 1}, 'C2': {(1, 1): 1}, 'C3': {(0, 0): 0.5, (1, 0): 0.5}}

    env = MultiContextEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, categories, feature_names, feature_values, feature_values_to_categories, probability_feature_values_in_categories)
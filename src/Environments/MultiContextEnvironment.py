from .Environment import *


class MultiContextEnvironment(Environment):
    """
    The MultiContextEnvironment class defines the advertising and pricing environment using, for each class, the models
    of:
    - the average dependence between the number of clicks and the bid;
    - the average cumulative daily click cost for the bid;
    - the conversion rate for 5 different prices.

    MultiContextEnvironment extends Environment.

    The MultiContextEnvironment class allows the agents to interact with it using its functions.
    From outside the structure of the context is not visible and the class allows to have an interaction with an
    environment with multiple contexts.

    Attributes:
        n_prices: Number of prices
        prices: Dictionary that maps each class of users to the values(price of the product) associated to the arms
        probabilities: Dictionary that maps each class to the bernoulli probabilities associated to the arms
        bids: Array of 100 possible bid values
        bids_to_clicks: Dictionary that maps each class to the parameters to build the function that models the number
            of clicks given the bid
        bids_to_clicks_variance: Variance of the gaussian noise associated to the function that models the number of
            clicks given the bid
        bids_to_cum_costs: Dictionary that maps each class to the parameters to build the function that models the
            cumulative daily click cost given the bid
        bids_to_cum_costs_variance: Variance of the gaussian noise associated to the function that models the cumulative
            daily click cost given the bid
        other_costs: Cost of the product
        categories: List containing the names of the possible categories the users can belong to
        feature_names: List containing the name of the features used to index the feature_values parameter
        feature_values: Dictionary containing the mapping between the features and the values the features can assume,
            the format is {feature_name: [value0, value1, value2, ...]}
        feature_values_to_categories: Dictionary containing the mapping between the features and the categories, the
            format is {tuple_of_features: category}
        probability_feature_values_in_categories: Dictionary giving the percentage of presence of a feature tuple inside
            each category, the format is {category: {tuple_of_features: probability}}
    """

    def __init__(self, n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, categories, feature_names, feature_values, feature_values_to_categories, probability_feature_values_in_categories):
        """
        Initializes the MultiContextEnvironment class

        :param int n_prices: Number of prices
        :param dict prices: Dictionary that maps each class of users to the values(price of the product) associated to
            the arms
        :param dict probabilities: Dictionary that maps each class to the bernoulli probabilities associated to the arms
        :param dict bids_to_clicks: Dictionary that maps each class to the parameters to build the function that models
            the number of clicks given the bid
        :param dict bids_to_cum_costs: Dictionary that maps each class to the parameters to build the function that
            models the cumulative daily click cost given the bid
        :param float other_costs: Cost of the product
        :param list categories: List containing the names of the possible categories the users can belong to
        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
            can assume, the format is {feature_name: [value0, value1, value2, ...]}
        :param dict feature_values_to_categories: Dictionary containing the mapping between the features and the
            categories, the format is {tuple_of_features: category}
        :param dict probability_feature_values_in_categories: Dictionary giving the percentage of presence of a feature
            tuple inside each category, the format is {category: {tuple_of_features: probability}}
        """

        super().__init__(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)

        self.categories = categories
        self.feature_name = feature_names
        self.feature_values = feature_values
        self.feature_values_to_categories = feature_values_to_categories
        self.probability_feature_values_in_categories = probability_feature_values_in_categories

    def get_reward(self, context, price_idx, conversion_prob, n_clicks, cum_daily_costs):
        """
        Returns the reward given the quantities used to compute it

        :param str context: Class of the user
        :param int price_idx: Index of the price
        :param np.ndarray conversion_prob: Conversion probability
        :param float n_clicks: Number of daily clicks
        :param float cum_daily_costs: Cumulative daily cost due to the advertising

        :returns: Reward
        :rtype: dict
        """

        return conversion_prob * (self.prices.get(self.feature_values_to_categories.get(context))[price_idx] - self.other_costs) * n_clicks - cum_daily_costs

    def round(self, price_idx, bid_idx, user_features_set):
        """
        Simulates a round in a pricing-advertising scenario, returning the realization of the chosen price, number of
        clicks and cumulative daily click cost given the price, the bid and features of the users for which the round
        has to be simulated

        :param int price_idx: Arm pulled in the current time step
        :param int bid_idx: Index of the bid used in the current round
        :param set tuple user_features_set: Tuple containing the features of the users for which the round has to be
            simulated

        :return: List of feature tuples for which the realizations are computed, Ordered realization of the pulled
            price, Ordered number of clicks, Ordered cumulative daily click cost
        :rtype: tuple
        """

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

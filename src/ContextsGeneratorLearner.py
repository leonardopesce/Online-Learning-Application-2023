from PricingAdvertisingLearner import PricingAdvertisingLearner
import itertools
class ContextGeneratorLearner:

    """
    Learner that tackles an environment with multiple contexts applying the multi-armed bandit algorithms using also a
    context generation algorithm

    Attributes:
        context: List containing the contexts the learner is using in the current time step
        learners: List containing a learner for each context that is applied to the relative context in the context list
        feature_name: List containing the name of the features used to index the feature_values parameter
        feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
    """
    def __init__(self, prices, bids, feature_names, feature_values, time_between_context_generation):
        """
        Initialize the multi-context learner

        :params numpy.ndarray prices: Prices in the pricing problem
        :params numpy.ndarray bids: Bids in the advertising problem
        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        :param int time_between_context_generation: Number of
        """
        # The Learner starts considering a single context with all the possible users inside
        self.context = list(self.create_user_feature_tuples(feature_names, feature_values))
        self.learners = list(PricingAdvertisingLearner(prices, bids))
        self.feature_to_observation = {feature_tuple: [] for feature_tuple in self.context}
        self.feature_name = feature_names
        self.feature_values = feature_values
        self.time_between_context_generation = time_between_context_generation

    def create_user_feature_tuples(self, feature_names, feature_values):
        """
        Creates all the possible combinations of the values of the features

        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}

        :returns: All the possible combinations of the values of the features
        :rtype: set
        """
        # Get the list of values for each feature
        value_lists = [feature_values[feature] for feature in feature_names]
        # Generate all possible combinations using itertools.product
        all_combinations = list(itertools.product(*value_lists))
        # Convert each combination to a tuple and add it to a set
        result_set = set(tuple(combination) for combination in all_combinations)

        return result_set

    def update_context(self):
        """
        Applies the context generation method in order to find a context structure that increases the value of the
        reward
        """

        # TODO probabilmente è necessario prendere tutte le osservazioni e ridistribuirle nei learners


    def pull_arm(self, other_costs):
        """
        Chooses the price to play and the bid for all the learners based on the learning algorithm of the learner

        :param float other_costs: Know costs of the product, used to compute the margin

        :return: Index of the price to pull, index of the bid to pull
        :rtype: tuple
        """
        pulled = []
        for idx in range(len(self.learners)):
            context_of_the_learner = self.context[idx]
            price_to_pull, bid_to_pull = self.learners[idx].pull_arm(other_costs)
            pulled.append([context_of_the_learner, price_to_pull, bid_to_pull])

        return pulled

    def update(self, features_list, pulled_price_list, bernoulli_realizations_list, pulled_bid_list, clicks_given_bid_list, c, rewards):
        """
        Updating the parameters of the learners based on the observations obtained by playing the chosen price and bid
        in the environment

        :param list features_list: Ordered sets of feature of the context of the learner that has given the observation
        :param list pulled_price_list: Prices pulled in the current time step by each learner
        :param list bernoulli_realizations_list: Bernoulli realizations of the pulled prices in the current time step by
        each learner
        :param list pulled_bid_list: Bids pulled in the current time step by each learner
        :param list clicks_given_bid_list: Number of clicks obtained playing the bid in the current time step by each
        learner
        :param list clicks_given_bid_list: Costs due to the advertising when playing the bid in the current time step by
        each learner
        :param list rewards: Rewards collected by each learner in the current time step playing the pulled arms
        """

        for idx in range(len(self.learners)):
            self.learners[idx].update(pulled_price_list[idx], bernoulli_realizations_list[idx], pulled_bid_list[idx], clicks_given_bid_list[idx], cost_given_bid_list[idx], rewards[idx])
            self.feature_to_observation[features_list[idx]].append([pulled_price_list[idx], bernoulli_realizations_list[idx], pulled_bid_list[idx], clicks_given_bid_list[idx], cost_given_bid_list[idx], rewards[idx]])
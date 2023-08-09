import itertools
import numpy as np

from PricingAdvertisingLearner import PricingAdvertisingLearner
from ContextLearner import ContextLearner
from LearnerFactory import LearnerFactory
from ContextTree.ContextTree import ContextTree


class ContextGeneratorLearner:

    """
    Learner that tackles an environment with multiple contexts applying the multi-armed bandit algorithms using also a
    context generation algorithm

    Attributes:
        contexts: List containing the contexts the learner is using in the current time step
        learners: List containing a learner for each context that is applied to the relative context in the context list
        feature_name: List containing the name of the features used to index the feature_values parameter
        feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
    """
    def __init__(self, prices, bids, feature_names, feature_values, time_between_context_generation, learner_type):
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
        self.t = 0
        self.prices = prices
        self.bids = bids
        self.context_learners = [ContextLearner(self.create_user_feature_tuples(feature_names, feature_values), LearnerFactory().get_learner(learner_type, prices, bids))]
        self.feature_to_observation = {feature_tuple: [] for feature_tuple in self.create_user_feature_tuples(feature_names, feature_values)}
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.time_between_context_generation = time_between_context_generation
        self.learner_type = learner_type

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
        result_set = set(all_combinations)

        return result_set

    def update_context1(self):
        context_tree = ContextTree(self.prices, self.bids, self.feature_names, self.feature_values, self.feature_to_observation, 0.95)
        new_contexts = context_tree.get_context_structure()
        print(new_contexts)
        # Redefining the learners to use in the next steps of the learning procedure using the new contexts
        # Defining the list of the new context learners (learners + contexts)
        new_learners = []
        # Iterating on the new contexts
        for context in new_contexts:
            # Defining a new learner
            new_learner = LearnerFactory().get_learner(self.learner_type, self.prices, self.bids)
            # Iterating on the tuples of features of the user in the context
            for feature_tuple in context:
                # Iterating on the observation regarding the user with the chosen values of features
                for element in self.feature_to_observation.get(feature_tuple):
                    # Updating the new learner using the past observation of the users in the context it has to consider
                    new_learner.update(element[0], element[1], element[2], element[3], element[4], element[5])

            # Appending a new context learner to the set of the new learner to use in future time steps
            new_learners.append(ContextLearner(context, new_learner))

        # Setting the new learners into the context generator learner
        self.context_learners = new_learners

    def pull_arm(self, other_costs: float) -> list[list[set, int, int]]:
        """
        Chooses the price to play and the bid for all the learners based on the learning algorithm of the learner

        :param float other_costs: Know costs of the product, used to compute the margin

        :return: Context of the Learner, Index of the price to pull, index of the bid to pull
        :rtype: list
        """
        pulled = []
        for learner in self.context_learners:
            context_of_the_learner = learner.get_context()
            price_to_pull, bid_to_pull = learner.get_learner().pull_arm(other_costs)
            pulled.append([context_of_the_learner, price_to_pull, bid_to_pull])

        return pulled

    def update(self, features_list, pulled_price_list, bernoulli_realizations_list, pulled_bid_list, clicks_given_bid_list, cost_given_bid_list, rewards):
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
        self.t += 1
        for idx, learner in enumerate(self.context_learners):
            learner.get_learner().update(pulled_price_list[idx], np.concatenate(bernoulli_realizations_list[idx], axis=0),
                                         pulled_bid_list[idx], np.sum(clicks_given_bid_list[idx]),
                                         np.sum(cost_given_bid_list[idx]), np.sum(rewards[idx]))
            for i, feature in enumerate(features_list[idx]):
                self.feature_to_observation[feature].append([pulled_price_list[idx], bernoulli_realizations_list[idx][i], pulled_bid_list[idx], clicks_given_bid_list[idx][i], cost_given_bid_list[idx][i], rewards[idx][i]])

        # rewards = [[1,2,3,4]] ---> [[1,2],[3,4]]
    def get_pulled_prices(self):
        """
        Returns the sequence of pulled prices

        :returns: Ordered sequence of pulled prices
        :rtype: list
        """
        pulled_prices = []
        for learner in self.context_learners:
            pulled_prices.append(learner.get_learner().get_pulled_prices())

        return pulled_prices

    def get_pulled_bids(self):
        """
        Returns the sequence of pulled bids

        :returns: Ordered sequence of pulled bids
        :rtype: list
        """
        pulled_bids = []
        for learner in self.context_learners:
            pulled_bids.append(learner.get_learner().get_pulled_bids())

        return pulled_bids

    def get_collective_reward(self):
        """
        Returns the collective reward

        :returns: Collective reward
        :rtype: float
        """
        collective_reward = 0
        for learner in self.context_learners:
            collective_reward += learner.get_learner().get_reward()

        return collective_reward

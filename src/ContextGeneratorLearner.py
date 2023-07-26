import itertools
import numpy as np

from PricingAdvertisingLearner import PricingAdvertisingLearner
from ContextLearner import ContextLearner
from LearnerFactory import LearnerFactory


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
        result_set = set(tuple(combination) for combination in all_combinations)

        return result_set

    def update_context(self):
        """
        Applies the context generation method in order to find a context structure that increases the value of the
        reward
        """

        # TODO probabilmente è necessario prendere tutte le osservazioni e ridistribuirle nei learners
        # Saving the reward of the aggregate model. It will be used to compare the reward of the new context.
        lower_bound = lambda delta, Z : np.sqrt((-np.log(delta)) / (2 * Z))

        num_samples_split_0_x = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[0] == 0))
        num_samples_split_0_y = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[0] == 1))
        num_samples_split_1_x = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[1] == 0))
        num_samples_split_1_y = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[1] == 1))
        tot_num_samples = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys()))

        aggregate_reward = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys())) - lower_bound(0.05, tot_num_samples)
        reward_split_0_x = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[0] == 0)) - lower_bound(0.05, num_samples_split_0_x)
        reward_split_0_y = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[0] == 1)) - lower_bound(0.05, num_samples_split_0_y)
        reward_split_1_x = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[1] == 0)) - lower_bound(0.05, num_samples_split_1_x)
        reward_split_1_y = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[1] == 1)) - lower_bound(0.05, num_samples_split_1_y)

        # Calculating the probabilities of the contexts TODO: hanno senso i lower bound sulle probabilità? (non credo)
        probability_split_0_x = (num_samples_split_0_x / tot_num_samples) - lower_bound(0.05, num_samples_split_0_x)
        probability_split_0_y = (num_samples_split_0_y / tot_num_samples) - lower_bound(0.05, num_samples_split_0_y)
        probability_split_1_x = (num_samples_split_1_x / tot_num_samples) - lower_bound(0.05, num_samples_split_1_x)
        probability_split_1_y = (num_samples_split_1_y / tot_num_samples) - lower_bound(0.05, num_samples_split_1_y)

        lower_bound_reward_0 = probability_split_0_x * reward_split_0_x + probability_split_0_y * reward_split_0_y
        lower_bound_reward_1 = probability_split_1_x * reward_split_1_x + probability_split_1_y * reward_split_1_y

        split_0 = -1
        if lower_bound_reward_0 > aggregate_reward and lower_bound_reward_0 > lower_bound_reward_1:
            split_0 = 0
            split_1 = 1
        elif lower_bound_reward_1 > aggregate_reward:
            split_0 = 1
            split_1 = 0

        if split_0 != -1:
            num_samples_split_branch_1_x = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[split_0] == 0 and key[split_0] == 0))
            num_samples_split_branch_1_y = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[split_0] == 0 and key[split_0] == 1))
            tot_num_samples = sum(obj[3] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[split_0] == 0))

            reward_split_branch_1_x = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[split_0] == 0 and key[split_0] == 0)) - lower_bound(0.05, num_samples_split_branch_1_x)
            reward_split_branch_1_y = np.mean(obj[-1] for obj in (self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[split_0] == 0 and key[split_0] == 1)) - lower_bound(0.05, num_samples_split_branch_1_y)

            # Calculating the probabilities of the contexts TODO: hanno senso i lower bound sulle probabilità? (non credo)
            probability_split_0_x = (num_samples_split_0_x / tot_num_samples) - lower_bound(0.05, num_samples_split_0_x)
            probability_split_0_y = (num_samples_split_0_y / tot_num_samples) - lower_bound(0.05, num_samples_split_0_y)
            probability_split_1_x = (num_samples_split_1_x / tot_num_samples) - lower_bound(0.05, num_samples_split_1_x)
            probability_split_1_y = (num_samples_split_1_y / tot_num_samples) - lower_bound(0.05, num_samples_split_1_y)

            lower_bound_reward_0 = probability_split_0_x * reward_split_0_x + probability_split_0_y * reward_split_0_y
            lower_bound_reward_1 = probability_split_1_x * reward_split_1_x + probability_split_1_y * reward_split_1_y

            split_x = -1
            if lower_bound_reward_0 > aggregate_reward and lower_bound_reward_0 > lower_bound_reward_1:
                split_x = 0
            elif lower_bound_reward_1 > aggregate_reward:
                split_x = 1
        defined_contexts = set()

        # Redefining the learners to use in the next steps of the learning procedure using the new contexts
        # Defining the list of the new context learners (learners + contexts)
        new_learners = []
        # Iterating on the new contexts
        for context in defined_contexts:
            # Defining a new learner
            new_learner = LearnerFactory().get_learner(self.learner_type, self.prices, self.bids)
            # Iterating on the tuples of features of the user in the context
            for feature_tuple in context:
                # Iterating on the observation regarding the user with the chosen values of features
                for element in self.feature_to_observation(feature_tuple):
                    # Updating the new learner using the past observation of the users in the context it has to consider
                    new_learner.update(element[0], element[1], element[2], element[3], element[4], element[5])

            # Appending a new context learner to the set of the new learner to use in future time steps
            new_learners.append(ContextLearner(context, new_learner))

        # Setting the new learners into the context generator learner
        self.context_learners = new_learners

    def pull_arm(self, other_costs):
        """
        Chooses the price to play and the bid for all the learners based on the learning algorithm of the learner

        :param float other_costs: Know costs of the product, used to compute the margin

        :return: Context of the Learner, Index of the price to pull, index of the bid to pull
        :rtype: tuple
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

        for idx, learner in enumerate(self.context_learners):
            learner.get_learner().update(pulled_price_list[idx], bernoulli_realizations_list[idx], pulled_bid_list[idx], clicks_given_bid_list[idx], cost_given_bid_list[idx], rewards[idx])
            self.feature_to_observation[features_list[idx]].append([pulled_price_list[idx], bernoulli_realizations_list[idx], pulled_bid_list[idx], clicks_given_bid_list[idx], cost_given_bid_list[idx], rewards[idx]])
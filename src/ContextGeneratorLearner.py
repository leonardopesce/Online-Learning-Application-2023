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

        num_samples_split_0_x = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[0] == 0)
        num_samples_split_0_y = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[0] == 1)
        num_samples_split_1_x = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[1] == 0)
        num_samples_split_1_y = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[1] == 1)
        tot_num_samples = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key))

        aggregate_reward = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key)]) - lower_bound(0.05, tot_num_samples)
        reward_split_0_x = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[0] == 0]) - lower_bound(0.05, num_samples_split_0_x)
        reward_split_0_y = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[0] == 1]) - lower_bound(0.05, num_samples_split_0_y)
        reward_split_1_x = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[1] == 0]) - lower_bound(0.05, num_samples_split_1_x)
        reward_split_1_y = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[1] == 1]) - lower_bound(0.05, num_samples_split_1_y)

        # Calculating the probabilities of the contexts TODO: hanno senso i lower bound sulle probabilità? (non credo)
        probability_split_0_x = (num_samples_split_0_x / tot_num_samples) - lower_bound(0.05, num_samples_split_0_x)
        probability_split_0_y = (num_samples_split_0_y / tot_num_samples) - lower_bound(0.05, num_samples_split_0_y)
        probability_split_1_x = (num_samples_split_1_x / tot_num_samples) - lower_bound(0.05, num_samples_split_1_x)
        probability_split_1_y = (num_samples_split_1_y / tot_num_samples) - lower_bound(0.05, num_samples_split_1_y)

        lower_bound_reward_0 = probability_split_0_x * reward_split_0_x + probability_split_0_y * reward_split_0_y
        lower_bound_reward_1 = probability_split_1_x * reward_split_1_x + probability_split_1_y * reward_split_1_y

        # If splitting on the first feature is better than the aggregate model and it is better than splitting on the second feature, then split on the first feature.
        # If splitting on the second feature is better than the aggregate model and it is better than splitting on the first feature, then split on the second feature.
        split_0 = -1
        if lower_bound_reward_0 > aggregate_reward and lower_bound_reward_0 >= lower_bound_reward_1:
            # At this point we decided to discriminate on the first feature, so here we grouped (0,0) and (0,1) together and (1,0) and (1,1) together.
            split_0 = 0 # At level 0 we split on the first feature
            split_1 = 1 # At level 1 we split on the second feature
            aggregate_reward_first_split = lower_bound_reward_0
            lower_bound_left_branch = probability_split_0_x * reward_split_0_x
            lower_bound_right_branch = probability_split_0_y * reward_split_0_y
            tot_num_samples_first_split = num_samples_split_0_x + num_samples_split_0_y
            defined_contexts = [((0,0), (0,1)), ((1,0), (1,1))]
        elif lower_bound_reward_1 > aggregate_reward:
            # At this point we decided to discriminate on the second feature, so here we grouped (0,0) and (1,0) together and (0,1) and (1,1) together.
            split_0 = 1 # At level 0 we split on the second feature
            split_1 = 0 # At level 1 we split on the first feature
            aggregate_reward_first_split = lower_bound_reward_1
            lower_bound_left_branch = probability_split_1_x * reward_split_1_x
            lower_bound_right_branch = probability_split_1_y * reward_split_1_y
            tot_num_samples_first_split = num_samples_split_1_x + num_samples_split_1_y
            defined_contexts = [((0,0), (1,0)), ((0,1), (1,1))]
        else:
            # If the aggregate model is better than the two splits, then we do not split at all.
            defined_contexts = [((0, 0), (0, 1), (1, 0), (1, 1))]
           

        if split_0 != -1:
            # Now given the split we have to find the best split on the second level.
            # We have to find the best split on the second level for each of the two splits on the first level.
            # It can be either split one of the two contexts, split only one of them or not split at all and keep the division in 2 groups.
            num_samples_split_branch_1_x = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 0 and key[split_1] == 0)
            num_samples_split_branch_1_y = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 0 and key[split_1] == 1)
            num_samples_split_branch_2_x = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 1 and key[split_1] == 0)
            num_samples_split_branch_2_y = sum(obj[3] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 1 and key[split_1] == 1)
            
            # Now we compute the rewards of the 4 splits.
            reward_split_branch_1_x = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 0 and key[split_1] == 0]) - lower_bound(0.05, num_samples_split_branch_1_x)
            reward_split_branch_1_y = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 0 and key[split_1] == 1]) - lower_bound(0.05, num_samples_split_branch_1_y)
            reward_split_branch_2_x = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 1 and key[split_1] == 0]) - lower_bound(0.05, num_samples_split_branch_2_x)
            reward_split_branch_2_y = np.mean([obj[-1] for key in self.feature_to_observation.keys() for obj in self.feature_to_observation.get(key) if key[split_0] == 1 and key[split_1] == 1]) - lower_bound(0.05, num_samples_split_branch_2_y)

            # Calculating the probabilities of the contexts TODO: hanno senso i lower bound sulle probabilità? (non credo)
            probability_split_branch_1_x = (num_samples_split_branch_1_x / tot_num_samples_first_split) - lower_bound(0.05, num_samples_split_branch_1_x)
            probability_split_branch_1_y = (num_samples_split_branch_1_y / tot_num_samples_first_split) - lower_bound(0.05, num_samples_split_branch_1_y)
            probability_split_branch_2_x = (num_samples_split_branch_2_x / tot_num_samples_first_split) - lower_bound(0.05, num_samples_split_branch_2_x)
            probability_split_branch_2_y = (num_samples_split_branch_2_y / tot_num_samples_first_split) - lower_bound(0.05, num_samples_split_branch_2_y)

            lower_bound_reward_branch_1_x = probability_split_branch_1_x * reward_split_branch_1_x
            lower_bound_reward_branch_1_y = probability_split_branch_1_y * reward_split_branch_1_y
            lower_bound_reward_branch_2_x = probability_split_branch_2_x * reward_split_branch_2_x
            lower_bound_reward_branch_2_y = probability_split_branch_2_y * reward_split_branch_2_y

            # Finally we check which is the best split on the second level. Here we can either split as follows:
            # - (0,0) and (0,1) together and (1,0), (1,1) splitted. (e.g. keep together the males and separate young females from old females)
            # - (0,0), (0,1) splitted and (1,0) and (1,1) together. (e.g. separate young males from old males and keep together the females)
            # - (0,0), (0,1), (1,0), (1,1) all splitted (i.e. perfect discrimination)
            # This holds if we splitted on the first feature during the first split. Otherwise we would have the following situations:
            # - (0,0) and (1,0) together and (0,1), (1,1) splitted. (e.g. keep together young people and discriminate between old males and old females).
            # - (0,0), (1,0) splitted and (0,1) and (1,1) together. (e.g. discriminate young males from young females and keep together old people).
            # - (0,0), (0,1), (1,0), (1,1) all splitted (i.e. perfect discrimination)
            reward_full_discrimination = lower_bound_reward_branch_1_x + lower_bound_reward_branch_1_y + lower_bound_reward_branch_2_x + lower_bound_reward_branch_2_y
            reward_left_aggr_right_discr = lower_bound_left_branch + lower_bound_reward_branch_2_x + lower_bound_reward_branch_2_y
            reward_left_discr_right_aggr = lower_bound_reward_branch_1_x + lower_bound_reward_branch_1_y + lower_bound_right_branch
            if reward_left_discr_right_aggr > aggregate_reward_first_split and reward_left_discr_right_aggr >= reward_full_discrimination:
                # If we firstly split on the first feature, it follows this grouping ((0,0), (0,1), ((1,0), (1,1)))
                # Otherwise if we firstly split on the second feature, it follows this grouping ((0,0), (1,0), ((0,1), (1,1)))
                if split_0 == 0:
                    defined_contexts = [((0,0),), ((0,1),), ((1,0), (1,1))]
                else:
                    defined_contexts = [((0,0),), ((1,0),), ((0,1), (1,1))]
            elif reward_left_aggr_right_discr > aggregate_reward_first_split and reward_left_aggr_right_discr >= reward_full_discrimination:
                # If we firstly split on the first feature, it follows this grouping (((0,0), (0,1)), (1,0), (1,1))
                # Otherwise if we firstly split on the second feature, it follows this grouping (((0,0), (1,0)), (0,1), (1,1))
                if split_0 == 0:
                    defined_contexts = [((0,0), (0,1)), ((1,0),), ((1,1),)]
                else:
                    defined_contexts = [((0,0), (1,0)), ((0,1),), ((1,1),)]
            elif reward_full_discrimination > aggregate_reward_first_split:
                print("sono qui porcoddio")
                defined_contexts = [((0,0),), ((0,1),), ((1,0),), ((1,1),)]

        # Redefining the learners to use in the next steps of the learning procedure using the new contexts
        # Defining the list of the new context learners (learners + contexts)
        print(defined_contexts)
        new_learners = []
        # Iterating on the new contexts
        for context in defined_contexts:
            # Defining a new learner
            new_learner = LearnerFactory().get_learner(self.learner_type, self.prices, self.bids)
            # Iterating on the tuples of features of the user in the context
            for feature_tuple in context:
                # Iterating on the observation regarding the user with the chosen values of features
                # print(context)
                for element in self.feature_to_observation.get(feature_tuple):
                    # Updating the new learner using the past observation of the users in the context it has to consider
                    new_learner.update(element[0], element[1], element[2], element[3], element[4], element[5])
            new_learner.t = self.t #TODO: check if this is correct, i.e. if it is needed to set the new learner time to the current timestep.
            # Appending a new context learner to the set of the new learner to use in future time steps
            new_learners.append(ContextLearner(context, new_learner))

        # Setting the new learners into the context generator learner.
        self.context_learners = new_learners

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
            new_learner.t = self.t #TODO: check if this is correct, i.e. if it is needed to set the new learner time to the current timestep.
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
        self.t += 1
        for idx, learner in enumerate(self.context_learners):
            learner.get_learner().update(pulled_price_list[idx], np.concatenate(bernoulli_realizations_list[idx], axis=0),
                                         pulled_bid_list[idx], np.sum(clicks_given_bid_list[idx]),
                                         np.sum(cost_given_bid_list[idx]), rewards[idx])
            for i in range(len(features_list[idx])):
                self.feature_to_observation[features_list[idx][i]].append([pulled_price_list[idx], bernoulli_realizations_list[idx][i], pulled_bid_list[idx], clicks_given_bid_list[idx][i], cost_given_bid_list[idx][i], rewards[idx]])


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

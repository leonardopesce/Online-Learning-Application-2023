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
    def __init__(self, prices, bids, feature_names, feature_values, time_between_context_generation, learner_type, other_costs):
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
        self.context_tree = None
        self.other_costs = other_costs

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

    def update_context(self):
        # To comment for run everything
        ###########
        for learner in self.context_learners:
            learner.get_learner().advertising_learner.plot_clicks()
        ###########
        print(f"I'm {self.learner_type} and I'm updating the context")

        old_context = None
        if self.context_tree is None:
            self.context_tree = ContextTree(self.prices, self.bids, self.feature_names, self.feature_values, self.feature_to_observation, 0.05, self.other_costs)
        else:
            old_context = self.context_tree.get_context_structure()
            self.context_tree = ContextTree(self.prices, self.bids, self.feature_names, self.feature_values, self.feature_to_observation, 0.05, self.other_costs)

        new_contexts = self.context_tree.get_context_structure()
        if old_context != new_contexts:
            print(f"{self.t} - {self.learner_type} - New Context: {new_contexts}")
            # Redefining the learners to use in the next steps of the learning procedure using the new contexts
            # Defining the list of the new context learners (learners + contexts)
            new_learners = []
            # Iterating on the new contexts
            for context in new_contexts:
                # Defining a new learner
                new_learner = LearnerFactory().get_learner(self.learner_type, self.prices, self.bids)
                # Refactoring the feature to observations dictionary
                #new_learner_feature_to_obs = {tuple(context): {}}
                new_learner_feature_to_obs = {}
                # Iterating on the tuples of features of the user in the context
                for feature_tuple in context:
                    reward_of_context = []  # {((0,1), (0,0)) : {(0,1): [1,2,3,4], (0,0): [1,2,3,4]}}
                    # Iterating on the observation regarding the user with the chosen values of features
                    new_learner_feature_to_obs[feature_tuple] = self.feature_to_observation.get(feature_tuple)
                    # for element in self.feature_to_observation.get(feature_tuple):
                    # Updating the new learner using the past observation of the users in the context it has to consider
                    # new_learner_feature_to_obs.get(context).append([element])
                    # new_learner.update(element[0], element[1], element[2], element[3], element[4], element[5])

                for key in new_learner_feature_to_obs.keys():
                    context_copy = context.copy()
                    context_copy.remove(key)
                    #mean_clicks = np.sum([np.mean([observation[3] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])
                    #mean_costs = np.sum([np.mean([observation[4] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])

                    #mean_rewards = np.sum([np.mean([observation[5] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])
                    # rewards = np.sum(np.array([[observation[5] for observation in new_learner_feature_to_obs.get(key)] for key in new_learner_feature_to_obs.keys()]), axis=0)

                    mean_clicks_per_feature_tuple = np.array([np.mean([observation[3] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])
                    #mean_costs_per_feature_tuple = np.array([np.mean([observation[4] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])
                    mean_clicks = np.sum([np.mean([observation[3] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])
                    mean_costs = np.sum([np.mean([observation[4] for observation in new_learner_feature_to_obs.get(sub_context)]) for sub_context in context_copy])

                    bernoulli_means_per_feature_tuple = np.zeros((len(mean_clicks_per_feature_tuple), len(self.prices)))

                    for sub_context_idx, sub_context in enumerate(context_copy):
                        observation_list = new_learner_feature_to_obs.get(sub_context)
                        bernoulli_obs_per_price = [np.array([]) for price in self.prices]

                        for observation in observation_list:
                            bernoulli_obs_per_price[observation[0]] = np.concatenate((bernoulli_obs_per_price[observation[0]], observation[1]))

                        bernoulli_means_per_price = np.array([np.mean(bernoulli_price_aggregated) for bernoulli_price_aggregated in bernoulli_obs_per_price])
                        bernoulli_means_per_price = np.nan_to_num(bernoulli_means_per_price)

                        bernoulli_means_per_feature_tuple[sub_context_idx, :] = bernoulli_means_per_price[:]

                    number_of_ones = np.floor(mean_clicks_per_feature_tuple[None, :] @ bernoulli_means_per_feature_tuple).astype(int)
                    number_of_zeros = np.ceil(mean_clicks_per_feature_tuple[None, :] @ (1 - bernoulli_means_per_feature_tuple)).astype(int)

                    for idx, observation in enumerate(new_learner_feature_to_obs.get(key)):
                        bernoulli_obs_other_classes = np.ones(number_of_ones[0, observation[0]])
                        bernoulli_obs_other_classes = np.concatenate((bernoulli_obs_other_classes, np.zeros(number_of_zeros[0, observation[0]])))
                        #print(idx)
                        #print(observation[1])
                        #print()
                        #print(bernoulli_obs_other_classes)
                        #print()
                        #print(np.concatenate((observation[1], bernoulli_obs_other_classes)))
                        #print("------------------------------------------------------------------------------------------------------------------")
                        new_learner.update(observation[0], np.concatenate((observation[1], bernoulli_obs_other_classes)), observation[2], observation[3] + mean_clicks, observation[4] + mean_costs, observation[5]) # rewards[idx])


                #flatten_obs = self.get_flattened_obs(new_learner_feature_to_obs, self.t)
                #for day in flatten_obs.get(tuple(context)):
                #    new_learner.update(day[0][0], day[0][1], day[0][2], day[0][3], day[0][4], day[0][5])

                new_learner.t = self.t
                # Appending a new context learner to the set of the new learner to use in future time steps
                new_learners.append(ContextLearner(context, new_learner))

            # Setting the new learners into the context generator learner
            self.context_learners = new_learners

        else:
            print("No changes in the context structure")

    def update_context1(self):
        print("Updating the context")

        old_context = None
        if self.context_tree is None:
            self.context_tree = ContextTree(self.prices, self.bids, self.feature_names, self.feature_values, self.feature_to_observation, 0.05)
        else:
            old_context = self.context_tree.get_context_structure()
            self.context_tree.split_leaves(self.feature_to_observation)

        new_contexts = self.context_tree.get_context_structure()
        if old_context != new_contexts:
            print(f"{self.t} - {self.learner_type} - New Context: {new_contexts}")
            # Redefining the learners to use in the next steps of the learning procedure using the new contexts
            # Defining the list of the new context learners (learners + contexts)
            new_learners = []
            # Iterating on the new contexts
            for context in new_contexts:
                # Defining a new learner
                new_learner = LearnerFactory().get_learner(self.learner_type, self.prices, self.bids)
                # Refactoring the feature to observations dictionary
                new_learner_feature_to_obs = {tuple(context): {}}
                # Iterating on the tuples of features of the user in the context
                for feature_tuple in context:
                    reward_of_context = [] # {((0,1), (0,0)) : {(0,1): [1,2,3,4], (0,0): [1,2,3,4]}}
                    # Iterating on the observation regarding the user with the chosen values of features
                    new_learner_feature_to_obs[tuple(context)][feature_tuple] = self.feature_to_observation.get(feature_tuple)
                    # for element in self.feature_to_observation.get(feature_tuple):
                        # Updating the new learner using the past observation of the users in the context it has to consider
                        # new_learner_feature_to_obs.get(context).append([element])
                        # new_learner.update(element[0], element[1], element[2], element[3], element[4], element[5])
                flatten_obs = self.get_flattened_obs(new_learner_feature_to_obs, self.t)
                for day in flatten_obs.get(tuple(context)):
                    new_learner.update(day[0][0], day[0][1], day[0][2], day[0][3], day[0][4], day[0][5])
                assert new_learner.t == self.t
                # Appending a new context learner to the set of the new learner to use in future time steps
                new_learners.append(ContextLearner(context, new_learner))

            # Setting the new learners into the context generator learner
            self.context_learners = new_learners
        else:
            print("No changes in the context structure")

    def get_flattened_obs(self, feature_to_obs, num_obs):
        # feature_to_obs is a dictionary of the form {context: {feature_tuple: [observation]}} where observation is a
        # list itself with 6 elements. We can have multiple observation for each feature_tuple but the list of
        # observation is the same for each feature_tuple in the context. We want to flatten the list of observation
        # in a single list of observation for each context.
        # For each observation of each tuple of features in the context, the function builds a combined observation
        # by doing the following operations:
        # 1. the first element of the new observation is exactly one of the two elements of the parent observations
        # 2. the second element of the new observation is the concatenation of the second elements of the parent observations
        # 3. the third element of the new observation is exactly one of the third elements of the parent observations
        # 4. the fourth element of the new observation is the sum of the fourth elements of the parent observations
        # 5. the fifth element of the new observation is the sum of the fifth elements of the parent observations
        # 6. the sixth element of the new observation is the sum of the sixth elements of the parent observations.
        # The function returns a dictionary of the form {context: [observation]} where observation is a list of 6 elements
        # with the observation being the one just built.

        # Defining the dictionary to return
        result = {}
        # Iterating on the contexts
        for context in feature_to_obs.keys():
            # Defining the list of observations to return
            obs_list = []
            # Iterating on the tuples of features in the context
            grouped_obs = []
            for i in range(num_obs):
                obs_of_day = []
                for feature_tuple in feature_to_obs.get(context).keys():
                    # Iterating on the observations of the tuple of features
                    obs_of_day.append(feature_to_obs.get(context).get(feature_tuple)[i])
                grouped_obs.append(obs_of_day)

            assert len(grouped_obs) == num_obs

            for daily_obs in grouped_obs:
                new_obs = []
                new_pulled_price = None
                new_bernoulli_realization = np.array([])
                new_pulled_bid = None
                new_clicks_given_bid = 0
                new_cost_given_bid_list = 0
                new_reward = 0

                for obs in daily_obs:
                    new_pulled_price = obs[0]
                    new_bernoulli_realization = np.concatenate((new_bernoulli_realization, obs[1]))
                    new_pulled_bid = obs[2]
                    new_clicks_given_bid += obs[3]
                    new_cost_given_bid_list += obs[4]
                    new_reward += obs[5]

                new_obs.append([new_pulled_price, new_bernoulli_realization, new_pulled_bid, new_clicks_given_bid, new_cost_given_bid_list, new_reward])
                obs_list.append(new_obs)
            # Appending the list of observations to the dictionary to return
            result[context] = obs_list
        return result

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
        Returns the collective reward obtained by the learner

        :returns: Collective reward
        :rtype: float
        """

        rewards = [[observation[-1] for observation in self.feature_to_observation.get(key)] for key in self.feature_to_observation.keys()]
        collective_reward = np.sum(np.array(rewards), axis=0)

        return collective_reward

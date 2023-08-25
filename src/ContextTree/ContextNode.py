import numpy as np
import torch
import matplotlib.pyplot as plt

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

from GPs import BaseGaussianProcess


def lower_bound1(delta, num_samples, interval):
    return np.sqrt((-np.log(delta) * (interval ** 2)) / (2 * num_samples))

def lower_bound(delta, num_samples):
    """
    Computes the lower bound used by the context generation algorithm

    :param float confidence: Confidence to use in the lower bound used in the context generation algorithm
    :param float num_samples: Number of samples in the context to evaluate

    :return: Lower bound used by the context generation algorithm
    :rtype: float
    """

    return np.sqrt((-np.log(delta)) / (2 * num_samples))


def remove_element_from_tuple(input_tuple, index):
    """
    Removes an element at a specific index from a tuple

    :params tuple input_tuple: Tuple to which an element has to be removed
    :params int index: Index of the element to remove

    :return: New tuple without the element to be removed
    :rtype: tuple
    """

    return input_tuple[:index] + input_tuple[index+1:]


class ContextNode:
    """
    Node of the context tree that is used to apply the context generation algorithm

    Attributes:
        feature_names: List containing the name of the features used to index the feature_values parameter
        feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        feature_to_observation: Dictionary of the observations divided by tuples of features, the format is
        {tuple_of_features: [observation_list_1, observation_list_2, ...]}
        confidence: Confidence to use in the lower bound used in the context generation algorithm
        aggregate_reward: Aggregate reward of the current context without splitting the node
        father: Father node
        children: List of child nodes
        choice: Name of the feature that the context generation algorithm decided to split in the current node, it is
        None if no context disaggregation is done
    """

    def __init__(self, prices, bids, whole_feature_names, feature_names, feature_values, feature_to_observation, confidence, father, other_costs):
        """
        Initializes the node of the context tree
        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        feature_to_observation: Dictionary of the observations divided by tuples of features, the format is
        {tuple_of_features: [observation_list_1, observation_list_2, ...]}
        :param float confidence: Confidence to use in the lower bound used in the context generation algorithm
        :param ContextNode father: Father node
        """

        self.prices = prices
        self.bids = bids
        self.whole_feature_names = whole_feature_names # ['age', 'sex']
        self.feature_names = feature_names # ['age', 'sex'] --> ['age'] / ['sex']
        self.feature_values = feature_values
        self.feature_to_observation = feature_to_observation
        self.confidence = confidence
        self.aggregate_reward = 0  # self.compute_aggregate_reward()
        self.father = father
        self.children = {}
        self.choice = None

        bernoulli_distributions_lists = {}
        for idx in range(len(prices)):
            bernoulli_distributions_lists[idx] = []
        for key in self.feature_to_observation.keys():
            for observation in self.feature_to_observation[key]:
                bernoulli_distributions_lists[observation[0]].extend(observation[1])

        bernoulli_distributions_est = []
        bernoulli_distributions_lower_bound = []
        for price in bernoulli_distributions_lists.keys():
            mean = np.mean(bernoulli_distributions_lists[price]) if len(bernoulli_distributions_lists[price]) != 0 else 0.0
            bernoulli_distributions_est.append(mean)
            lower_bound = np.sqrt(-np.log(self.confidence) / 2 / len(bernoulli_distributions_lists[price])) if len(bernoulli_distributions_lists[price]) != 0 else 0.0
            bernoulli_distributions_lower_bound.append(lower_bound)

        bids_obs = [] # [1, 4, 5]
        clicks_obs = [] # [3235434243, 23423424, 24324234]
        costs_obs = [] # [4234243, 61423424, 122423.31]

        for key in self.feature_to_observation.keys():
            for observation in self.feature_to_observation[key]:
                bids_obs.append(self.bids[observation[2]])
                clicks_obs.append(observation[3])
                costs_obs.append(observation[4])

        kernel_clicks = ScaleKernel(RBFKernel())
        kernel_costs = ScaleKernel(RBFKernel())
        likelihood_clicks = GaussianLikelihood()
        likelihood_costs = GaussianLikelihood()
        self.gp_clicks = BaseGaussianProcess(likelihood=likelihood_clicks, kernel=kernel_clicks)
        self.gp_costs = BaseGaussianProcess(likelihood=likelihood_costs, kernel=kernel_costs)

        self.gp_clicks.fit(torch.Tensor(bids_obs), torch.Tensor(clicks_obs))
        self.gp_costs.fit(torch.Tensor(bids_obs), torch.Tensor(costs_obs))
        self.means_clicks, self.variance_clicks, self.lower_bounds_clicks, self.upper_bounds_clicks = self.gp_clicks.predict(torch.Tensor(self.bids))
        self.means_costs, self.variance_costs, self.lower_bounds_costs, self.upper_bounds_costs = self.gp_costs.predict(torch.Tensor(self.bids))

        # Note that price_obs and bids_obs are the indices of the prices and bids that are used in the GP.
        # They need to be converted to the actual prices and bids.
        self.means_rewards = None
        self.sigmas_rewards = None
        self.lower_bounds_rewards = None
        self.upper_bounds_rewards = None

        self.other_costs = other_costs
        rewards = (self.means_clicks[:, None] - np.sqrt(self.variance_clicks)[:, None]) * (np.array(bernoulli_distributions_est)[None, :] - np.array(bernoulli_distributions_lower_bound)[None, :]) * (self.prices - self.other_costs) - (self.means_costs[:, None] + np.sqrt(self.variance_costs)[:, None])
        #print(np.max(rewards))
        self.aggregate_reward = np.max(rewards)

    def __init__2(self, prices, bids, whole_feature_names, feature_names, feature_values, feature_to_observation, confidence, father):
        """
        Initializes the node of the context tree
        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        feature_to_observation: Dictionary of the observations divided by tuples of features, the format is
        {tuple_of_features: [observation_list_1, observation_list_2, ...]}
        :param float confidence: Confidence to use in the lower bound used in the context generation algorithm
        :param ContextNode father: Father node
        """

        self.prices = prices
        self.bids = bids
        self.whole_feature_names = whole_feature_names # ['age', 'sex']
        self.feature_names = feature_names # ['age', 'sex'] --> ['age'] / ['sex']
        self.feature_values = feature_values
        self.feature_to_observation = feature_to_observation
        self.confidence = confidence
        self.aggregate_reward = 0  # self.compute_aggregate_reward()
        self.father = father
        self.children = {}
        self.choice = None

        kernel = ScaleKernel(RBFKernel())
        likelihood = GaussianLikelihood() # [[[5, [1,0,1,1,0], 3, 45, 5,4]], ...14..., []]

        self.gp_reward = BaseGaussianProcess(likelihood=likelihood, kernel=kernel)

        # Note that price_obs and bids_obs are the indices of the prices and bids that are used in the GP.
        # They need to be converted to the actual prices and bids.
        self.flattened_obs = self.get_flattened_observations()
        self.means_rewards = None
        self.sigmas_rewards = None
        self.lower_bounds_rewards = None
        self.upper_bounds_rewards = None

        num_samples = sum(len(key) for key in self.feature_to_observation.values())
        self.update_gp()

    def update_gp(self):
        self.flattened_obs = self.get_flattened_observations()
        price_obs = np.array([obs[0][0] for obs in self.flattened_obs])
        bids_obs = np.array([obs[0][2] for obs in self.flattened_obs])

        x = torch.Tensor(np.block([self.prices[price_obs][:, None], self.bids[bids_obs][:, None]]))

        y = torch.Tensor([obs[0][-1] for obs in self.flattened_obs])
        # y = (y - y.min()) / (y.max() - y.min())
        self.gp_reward.fit(x, y)

        # Create a vector with 2 columns such that the first column is the price and the second column is the bid.
        # Create all the possible combinations of price and bid.
        price_bids = torch.Tensor(np.array(np.meshgrid(self.prices, self.bids)).T.reshape(-1, 2))
        # x = torch.Tensor(np.block([self.prices[:, None], self.bids[:, None]]))
        self.means_rewards, self.sigmas_rewards, self.lower_bounds_rewards, self.upper_bounds_rewards = self.gp_reward.predict(price_bids)

        best_reward_idx = np.argmax(self.means_rewards)
        best_reward = self.means_rewards[best_reward_idx]

        #print(f"Maximum of the means rewards: {max(self.means_rewards)}")
        #print(f"Sigma of the maximum of the means rewards: {self.sigmas_rewards[best_reward_idx]}")

        # print(f'Lower bound {lower_bound1(self.confidence, num_samples, np.max(means_rewards) - np.min(means_rewards))}')

        self.aggregate_reward = best_reward - 2.5 * self.sigmas_rewards[best_reward_idx]

        # idx = np.argmax(means_rewards - sigmas_rewards)
        # self.aggregate_reward = np.max(means_rewards - sigmas_rewards)
        # print(lower_bounds_rewards)
        # self.aggregate_reward = np.max(lower_bounds_rewards)
        # self.aggregate_reward = np.max(means_rewards - lower_bound1(0.01, num_samples, 1)) # np.max(means_rewards - lower_bound1(self.confidence, num_samples, 1)) #np.max(means_rewards) - np.min(means_rewards)))

    def get_flattened_observations(self, fto=None):
        if fto is None:
            fto = self.feature_to_observation

        result = []
        temp = []
        for key in fto.keys():
            temp.append(fto.get(key))

        grouped_obs = []
        for i in range(len(temp[0])):
            daily_obs = []
            for j in range(len(temp)):
                daily_obs.append(temp[j][i])
            grouped_obs.append(daily_obs)

        for daily_obs in grouped_obs:
            new_obs = []
            new_pulled_price_id = None
            new_bernoulli_realizations = np.array([])
            new_pulled_bid_id = None
            new_clicks_given_bid = 0
            new_cost_given_bid = 0
            new_reward = 0

            for ob in daily_obs:
                new_pulled_price_id = ob[0]
                new_bernoulli_realizations = np.concatenate((new_bernoulli_realizations, ob[1]))
                new_pulled_bid_id = ob[2]
                new_clicks_given_bid += ob[3]
                new_cost_given_bid += ob[4]
                new_reward += ob[5]

            new_obs.append([new_pulled_price_id, new_bernoulli_realizations, new_pulled_bid_id, new_clicks_given_bid, new_cost_given_bid, new_reward])
            result.append(new_obs)

        return result



    def compute_aggregate_reward(self):
        """
        Computes the aggregate reward given the context of the current node

        :return: Value of the reward of the current aggregate context
        :rtype: float
        """

        # Computing the total number of observations considered in the node
        total_num_samples = sum(observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key))

        # Computing the lower bound of the reward of the aggregate context
        return np.mean([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)]) - lower_bound(0.05, total_num_samples)

    def split(self):
        """
        Splits the node and the observations in new child nodes in the case the condition of the context generation
        algorithm is verified
        """

        if len(self.feature_names) > 0:
            # Finding the attribute with the highest marginal increase

            # Computing the total number of observations considered in the node
            total_num_samples = sum([observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)])

            # Computing the lower bounds of the rewards of disaggregate contexts for a split on the various features
            # taken separately
            feature_split_to_children = {}
            rewards_sub_contexts = {}
            feature_values_to_reward = {}

            #print(f"The aggregate reward of the not split node is {self.aggregate_reward}")

            for feature_name in self.feature_names:
                #print(f"Testing the split on {feature_name}")
                feature_values_to_num_samples = {}
                feature_values_to_reward_probability_split = {}
                feature_values_to_reward_lower_bound = {}
                feature_idx = self.whole_feature_names.index(feature_name)
                feature_split_to_children[feature_name] = {}

                new_feature_names = self.feature_names.copy()
                new_feature_names.remove(feature_name)
                new_feature_values = self.feature_values.copy()
                new_feature_values.pop(feature_name)

                for feature_value in self.feature_values[feature_name]:
                    #print(f"Evaluating {feature_value}")
                    # Computing the number of samples when splitting on the feature with feature_name and considering
                    # to take the samples with value of feature_name equal to the value feature_value
                    feature_values_to_num_samples[feature_value] = sum([observation[3] for key in self.feature_to_observation.keys()  for observation in self.feature_to_observation.get(key) if key[feature_idx] == feature_value])

                    # Computing the lower bound of the reward given by the context obtained splitting on the feature
                    # with feature_name and considering to take the samples with value of feature_name equal to the
                    # value feature_value
                    child_observations = {key: self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[feature_idx] == feature_value}

                    child = ContextNode(self.prices, self.bids, self.whole_feature_names, new_feature_names,
                                        new_feature_values, child_observations, self.confidence, self, self.other_costs)

                    feature_split_to_children[feature_name][feature_value] = child # {sex : { 0 : nodo femmine, 1 : nodo maschi }, age : { 0 : nodo giovani, 1 : nodo vecchi }}

                    feature_values_to_reward_lower_bound[feature_value] = child.aggregate_reward
                    # feature_values_to_reward_lower_bound[feature_value] = np.mean([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key) if key[feature_idx] == feature_value]) - lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

                    # Computing the lower bound of the probability of having the context obtained by splitting on the
                    # feature with feature_name and considering to take the samples with value of feature_name equal to
                    # the value feature_value
                    feature_values_to_reward_probability_split[feature_value] = (feature_values_to_num_samples[feature_value] / total_num_samples) - np.sqrt(-np.log(self.confidence) / 2 / feature_values_to_num_samples[feature_value]) #- lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

                # Computing the lower bound of the reward given by splitting on the feature with name feature_name as
                # the sum of the product between the reward of a context and the probability of that context
                feature_values_to_reward[feature_name] = sum([feature_values_to_reward_lower_bound[feature_value] * feature_values_to_reward_probability_split[feature_value] for feature_value in self.feature_values[feature_name]])
                #print(f"The reward coming from the split on {feature_name} is {feature_values_to_reward[feature_name]}")


            # Finding the name of the feature with the highest lower bound of the reward (feature on which the node
            # should possibly split)
            name_feature_max_reward = max(feature_values_to_reward, key=feature_values_to_reward.get)

            #print(f"Feature values to reward: {feature_values_to_reward}")

            # If the lower bound of the reward given by splitting in disaggregate context is higher than the reward of
            # the aggregate model in the node the context is split on the found feature
            if feature_values_to_reward[name_feature_max_reward] > self.aggregate_reward:
                print(f"Splitted on {name_feature_max_reward}" + f" the node with {self.father.choice} equal to {[item[0] for item in self.father.children.items() if item[1] == self][0]}" if self.father is not None else f"Splitted on {name_feature_max_reward}")

                # Setting the feature to use to separate the contexts
                self.choice = name_feature_max_reward

                # Setting the children of the node to the ones that guarantee a higher reward
                for feature_value in self.feature_values[name_feature_max_reward]:
                    self.children[feature_value] = feature_split_to_children[name_feature_max_reward][feature_value]

                # Running the creation of the subtree also on the children of the current node
                for child_key in self.children:
                    self.children[child_key].split()

    def get_contexts(self):
        """
        Computes recursively the context following the structure of the tree

        :return: Context of the subtree that starts from the node
        :rtype: list
        """

        contexts = []
        if len(self.children) != 0:
            for key_child in self.children:
                child_contexts = self.children[key_child].get_contexts()
                if len(child_contexts) != 0:
                    for context in child_contexts:
                        context[self.choice] = key_child

                        contexts.append(context.copy())
                else:
                    contexts.append({self.choice: key_child})

        return contexts

    def set_feature_to_observation(self, new_feature_to_obs: dict) -> None:
        self.feature_to_observation = new_feature_to_obs

    @property
    def leaves(self) -> list:
        leaves = []

        if len(self.children) != 0:
            for key_child in self.children:
                child_leaves = self.children[key_child].leaves
                leaves.extend(child_leaves)
        else:
            leaves.append(self)

        return leaves

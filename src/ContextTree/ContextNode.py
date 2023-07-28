import numpy as np
import torch

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

from GPs import BaseGaussianProcess



def lower_bound(confidence, num_samples):
    """
    Computes the lower bound used by the context generation algorithm

    :param float confidence: Confidence to use in the lower bound used in the context generation algorithm
    :param float num_samples: Number of samples in the context to evaluate

    :return: Lower bound used by the context generation algorithm
    :rtype: float
    """

    return np.sqrt((-np.log(confidence)) / (2 * num_samples))


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
        expanded: True, if the node has been evaluated by the context generation algorithm; False otherwise
    """

    def __init__(self, prices, bids, feature_names, feature_values, feature_to_observation, confidence, father):
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
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_to_observation = feature_to_observation
        self.confidence = confidence
        self.aggregate_reward = 0#self.compute_aggregate_reward()
        self.father = father
        self.children = {}
        self.choice = None
        self.expanded = False

        kernel = ScaleKernel(RBFKernel())
        likelihood = GaussianLikelihood()

        self.gp_reward = BaseGaussianProcess(likelihood=likelihood, kernel=kernel)

        price_obs = np.array([observation[0] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)])
        bids_obs = np.array([observation[2] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)])

        x = torch.Tensor(np.block([price_obs[:, None], bids_obs[:, None]]))

        y = torch.Tensor([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)])

        self.gp_reward.fit(x, y)
        
        # Create a vector with 2 columns such that the first column is the price and the second column is the bid.
        # Create all the possible combinations of price and bid.
        price_bids = torch.Tensor(np.array(np.meshgrid(self.prices, self.bids)).T.reshape(-1, 2))
        # print(price_bids, price_bids.shape)
            
        # x = torch.Tensor(np.block([self.prices[:, None], self.bids[:, None]]))
        means_rewards, sigmas_rewards, lower_bounds_rewards, upper_bounds_rewards = self.gp_reward.predict(price_bids)

        num_samples = sum(observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key))
        self.aggregate_reward = np.max(lower_bounds_rewards) # np.max(means_rewards - sigmas_rewards) # * lower_bound(self.confidence, num_samples)

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
        if not self.expanded and len(self.feature_names) > 0:
            # Finding the attribute with the highest marginal increase

            # Computing the total number of observations considered in the node
            total_num_samples = sum(observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key))

            # Computing the lower bounds of the rewards of the disaggregate contexts for a split on the various features
            # taken separately
            rewards_sub_contexts = {}
            for feature_name in self.feature_names:
                feature_values_to_num_samples = {}
                feature_values_to_reward_probability_split = {}
                feature_values_to_reward = {}
                feature_values_to_reward_lower_bound = {}
                feature_idx = self.feature_names.index(feature_name)
                for feature_value in self.feature_values[feature_name]:
                    # Computing the number of samples when splitting on the feature with feature_name and considering
                    # to take the samples with value of feature_name equal to the value feature_value
                    feature_values_to_num_samples[feature_value] = sum(observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key) if key[feature_idx] == feature_value)

                    # Computing the lower bound of the reward given by the context obtained splitting on the feature
                    # with feature_name and considering to take the samples with value of feature_name equal to the
                    # value feature_value
                    child = ContextNode(self.prices, self.bids, self.feature_names, self.feature_values,
                                        {key : self.feature_to_observation.get(key) for key in self.feature_to_observation.keys() if key[feature_idx] == feature_value}, self.confidence, self)

                    feature_values_to_reward_lower_bound[feature_value] = child.aggregate_reward
                    # feature_values_to_reward_lower_bound[feature_value] = np.mean([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key) if key[feature_idx] == feature_value]) - lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

                    # Computing the lower bound of the probaility of having the context obtained by splitting on the
                    # feature with feature_name and considering to take the samples with value of feature_name equal to
                    # the value feature_value
                    feature_values_to_reward_probability_split[feature_value] = (feature_values_to_num_samples[feature_value] / total_num_samples) #- lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

                # Computing the lower bound of the reward given by splitting on the feature with name feature_name as
                # the sum of the product between the reward of a context and the probability of that context
                feature_values_to_reward[feature_name] = sum([feature_values_to_reward_lower_bound[feature_value] * feature_values_to_reward_probability_split[feature_value] for feature_value in self.feature_values[feature_name]])

            # Finding the name of the feature with the highest lower bound of the reward (feature on which the node
            # should possibly split)
            name_feature_max_reward = max(feature_values_to_reward, key=feature_values_to_reward.get)

            # If the lower bound of the reward given by splitting in disaggregate context is higher than the reward of
            # the aggregate model in the node the context is split on the found feature
            if feature_values_to_reward[name_feature_max_reward] > self.aggregate_reward:
                # Setting the feature to use to separate the contexts
                self.choice = name_feature_max_reward

                # Computing the new values to give to the new nodes and executing the split
                chosen_feature_idx = self.feature_names.index(name_feature_max_reward)
                new_feature_names = self.feature_names.copy()
                new_feature_names.remove(name_feature_max_reward)
                new_feature_values = self.feature_values.copy()
                new_feature_values.pop(name_feature_max_reward)

                for feature_value in self.feature_values[name_feature_max_reward]:
                    # Computing the new observations to give to the specific child keeping only the sets of observations
                    # respecting the condition that says that the observations are related to samples in that context:
                    # key_of_dictionary_of_observation[name_feature_max_reward] == feature_value
                    new_feature_to_observation = {remove_element_from_tuple(key, chosen_feature_idx): self.feature_to_observation[key] for key in self.feature_to_observation if key[chosen_feature_idx] == feature_value}

                    # Creating ContextNode children and initializing them
                    self.children[feature_value] = ContextNode(self.prices, self.bids, new_feature_names, new_feature_values, new_feature_to_observation, self.confidence, self)

                # Running the creation of the subtree also on the children of the current node
                for child_key in self.children:
                    self.children[child_key].split()

            # Setting the current node as expanded
            self.expanded = True

    def get_contexts(self):
        """
        Computes recursively the context following the structure of the three

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

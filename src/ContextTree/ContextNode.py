import numpy as np


def lower_bound(confidence, num_samples):
    return np.sqrt((-np.log(confidence)) / (2 * num_samples))


def remove_element_from_tuple(input_tuple, index):
    return input_tuple[:index] + input_tuple[index+1:]


class ContextNode:
    """

    Attributes:
        feature_names:
        feature_to_observation: contains only the observation useful for the context in the format
    """
    def __init__(self, feature_names, feature_values, feature_to_observation, confidence, father):
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_to_observation = feature_to_observation
        self.confidence = confidence
        self.aggregate_reward = self.compute_aggregate_reward()
        self.father = father
        self.children = {}
        self.choice = None
        self.expanded = False

    def compute_aggregate_reward(self):
        """

        :return:
        :rtype:
        """

        # Computing the total number of observations considered in the node
        total_num_samples = sum(observation[3] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key))

        # Computing the lower bound of the reward of the aggregate context
        return np.mean([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key)]) - lower_bound(0.05, total_num_samples)

    def split(self):
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
                    feature_values_to_reward_lower_bound[feature_value] = np.mean([observation[-1] for key in self.feature_to_observation.keys() for observation in self.feature_to_observation.get(key) if key[feature_idx] == feature_value]) - lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

                    # Computing the lower bound of the probaility of having the context obtained by splitting on the
                    # feature with feature_name and considering to take the samples with value of feature_name equal to
                    # the value feature_value
                    feature_values_to_reward_probability_split[feature_value] = (feature_values_to_num_samples[feature_value] / total_num_samples) - lower_bound(self.confidence, feature_values_to_num_samples[feature_value])

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
                    self.children[feature_value] = ContextNode(new_feature_names, new_feature_values, new_feature_to_observation, self.confidence, self)

                # Running the creation of the subtree also on the children of the current node
                for child_key in self.children:
                    self.children[child_key].split()

            # Setting the current node as expanded
            self.expanded = True

    def get_contexts(self):
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

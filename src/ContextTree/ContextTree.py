import itertools

from .ContextNode import ContextNode


class ContextTree:
    """
    Data structure used to apply the context generation in an automatic and recursive way

    Attributes:
        root: Root node of the tree from which the context generation starts
        feature_names: List containing the name of the features used to index the feature_values parameter
        feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        confidence: Confidence to use in the lower bound used in the context generation algorithm
        context_structure: Final context structure generated by the algorithm, it is set the first time the structure is
        requested
    """

    def __init__(self, prices: dict, bids, feature_names: list, feature_values: dict, feature_to_observation: dict, confidence: float):
        """
        Initialize the data structure

        :param list feature_names: List containing the name of the features used to index the feature_values parameter
        :param dict feature_values: Dictionary containing the mapping between the features and the values the features
        can assume, the format is {feature_name: [value0, value1, value2, ...]}
        :param dict feature_to_observation: Dictionary of the observations divided by tuples of features, the format is
        {tuple_of_features: [observation_list_1, observation_list_2, ...]}
        :param float confidence: Confidence to use in the lower bound used in the context generation algorithm
        """

        self.root = ContextNode(prices, bids, feature_names, feature_names, feature_values, feature_to_observation, confidence, None)
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.confidence = confidence
        self.context_structure = None
        self.create_context_structure()

    def create_context_structure(self):
        """
        Create the context structure given the observations and applying the context generation algorithm
        """

        self.root.split()

    def split_leaves(self, feature_to_obs: dict):
        """
        Starts the splitting procedure that applies the context generation algorithm starting from the leaf nodes.
        The new observations to give to the leaves are needed

        :param dict feature_to_obs: Dictionary of the new observations divided by tuples of features, the format is
        {tuple_of_features: [observation_list_1, observation_list_2, ...]}

        :returns:
        :rtype: dict
        """
        context_structure = self.get_context_structure()
        leaves = self.root.leaves

        assert len(context_structure) == len(leaves)

        for i, leaf in enumerate(leaves):
            new_feature_to_observation = {key: feature_to_obs.get(key) for key in feature_to_obs.keys() if key in context_structure[i]}
            leaf.set_feature_to_observation(new_feature_to_observation)
            leaf.split()


    def get_context_structure(self):
        """
        Returns the contexts generated by the context generation algorithm

        :return: dictionary containing hte tuples representing the contexts that the context generation algorithm
        created
        :rtype: dict
        """

        context_structure = self.root.get_contexts()
        context_structure_final = []

        if len(context_structure) == 0:
            # Get the list of values for each feature
            value_lists = [self.feature_values[feature] for feature in self.feature_names]
            # Generate all possible combinations using itertools.product
            all_combinations = list(itertools.product(*value_lists))
            # Convert each combination to a tuple and add it to a set
            context_structure_final.append(all_combinations)
        else:
            for context in context_structure:
                list_of_tuples = [()]
                for feature_name in self.feature_names:
                    if feature_name in context.keys():
                        # Iterating on the tuples of features composing the context e.g. {feature2: 1, feature3: 1} --> [(0,1,1),(1,1,1)] --> [(0,1), (1,1)]
                        for i, context_tuple in enumerate(list_of_tuples):
                            list_of_tuples[i] = context_tuple + tuple([context[feature_name]])
                    else:
                        # Iterating on the tuples of features composing the context e.g. {feature2: 1, feature3: 1} --> [(0,1,1),(1,1,1)] --> [(0,1), (1,1)]
                        new_list_of_tuples = []
                        for context_tuple in list_of_tuples:
                            for feature_val in self.feature_values[feature_name]:
                                new_list_of_tuples.append(context_tuple + tuple([feature_val]))
                        list_of_tuples = new_list_of_tuples

                context_structure_final.append(list_of_tuples)

        self.context_structure = context_structure_final

        return context_structure_final

if __name__ == '__main__':
    feature_names = ['age', 'sex']
    feature_values = {'age': [0, 1], 'sex': [0, 1]}
    feature_to_observation = {(0, 0): [[0, 0, 0, 40, 0, 541]],
                              (0, 1): [[0, 0, 0, 54, 0, 541], [0, 0, 0, 123, 0, 4234], [0, 0, 0, 43, 0, 344]],
                              (1, 0): [[0, 0, 0, 4, 0, 17], [0, 0, 0, 44, 0, 541], [0, 0, 0, 64, 0, 333], [0, 0, 0, 46, 0, 312]],
                              (1, 1): [[0, 0, 0, 54, 0, 541]]}
    confidence = 0
    tree = ContextTree(feature_names, feature_values, feature_to_observation, confidence)
    print(tree.get_context_structure())
    print(f'The root splits on {tree.root.choice if tree.root.choice is not None else "nothing"}')

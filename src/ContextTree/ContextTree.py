from ContextNode import ContextNode


class ContextTree:

    def __init__(self, feature_names, feature_values, feature_to_observation):
        self.root = ContextNode(feature_names, feature_values, feature_to_observation)
        self.feature_names = feature_names
        self.feature_values = feature_values


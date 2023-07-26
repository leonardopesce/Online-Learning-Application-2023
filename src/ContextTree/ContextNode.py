
class ContextNode:
    def __init__(self, feature_names, feature_values, feature_to_observation):
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.children = []
        self.expanded = False
        self.feature_to_observation = feature_to_observation

    def split(self):
        if not self.expanded:
            chosen_feature = 1
            for feature_valaue in self.feature_values[chosen_feature]:
                self.children.append(ContextNode())
            self.expanded = True

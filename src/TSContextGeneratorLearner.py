from ContextGeneratorLearner import ContextGeneratorLearner

class TSContextGeneratorLearner(ContextGeneratorLearner):

    def __init__(self, prices, bids, feature_names, feature_values, time_between_context_generation):
        super().__init__(prices, bids, feature_names, feature_values, time_between_context_generation, "TS")
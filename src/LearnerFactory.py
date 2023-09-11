from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising

class LearnerFactory:
    def __init__(self) -> None:
        pass

    def get_learner(self, learner_type, prices, bids):
        # Check whether the type of Learner is TS or UCB and create the corresponding learner
        if learner_type == "TS":
            return TSLearnerPricingAdvertising(prices, bids)
        elif learner_type == "UCB":
            return UCBLearnerPricingAdvertising(prices, bids)
        else:
            raise ValueError("Learner type not supported")
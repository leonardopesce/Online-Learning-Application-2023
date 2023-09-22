from .TSPricingAdvertising import TSLearnerPricingAdvertising
from .UCBPricingAdvertising import UCBLearnerPricingAdvertising


class LearnerFactory:
    """
    Factory class to create the learners
    """

    def __init__(self) -> None:
        pass

    def get_learner(self, learner_type, prices, bids):
        """
        Creates the learner based on the type requested

        :params str learner_type: Type of learner to create
        :params numpy.ndarray prices: Prices in the pricing problem
        :params numpy.ndarray bids: Bids in the advertising problem

        :return: Learner created
        :rtype: PricingAdvertisingLearner
        """

        # Check whether the type of Learner is TS or UCB and create the corresponding learner
        if learner_type == "TS":
            return TSLearnerPricingAdvertising(prices, bids)
        elif learner_type == "UCB":
            return UCBLearnerPricingAdvertising(prices, bids)
        else:
            raise ValueError("Learner type not supported")

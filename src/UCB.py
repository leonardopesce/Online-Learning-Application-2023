import numpy as np
from Learner import *


class UCBLearner(Learner):
    """
    Learner that applies the Upper Confidence 1 algorithm.
    self.empirical_means: empirical means.
    self.times_arms_played: number of times that an arm has been played.
    self.confidence: upper confidence interval.
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.times_arms_played = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        """
        Chooses the arm to play based on the Upper Confidence 1 algorithm, therefore choosing the arm with higher upper
        confidence bound, which is the sum of the mean of the reward of the arm plus a confidence interval.

        :return: index of the arm to pull.
        """

        upper_confidence_bound = self.empirical_means + self.confidence
        idx = np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

        return idx

    def update(self, pulled_arm, reward):
        """
        Updating given the observations of the results obtained by playing the
        pulled arm in the environment.

        :param pulled_arm: arm pulled in the current time step.
        :param reward: reward collected in the current time step playing the pulled arm.
        """

        self.t += 1
        self.times_arms_played[pulled_arm] += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.times_arms_played[pulled_arm] - 1) + reward) / self.times_arms_played[pulled_arm]

        for arm in range(self.n_arms):
            self.confidence[arm] = (2 * np.log(self.t) / self.times_arms_played[arm]) ** 0.5 if self.times_arms_played[arm] > 0 else np.inf

        self.update_observations(pulled_arm, reward)

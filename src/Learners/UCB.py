import numpy as np

from .Learner import *


class UCBLearner(Learner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) algorithm

    Attributes:
        successes_per_arm: Number of times each arm has obtained a positive realization
        total_observations_per_arm: Total number of realizations of each arm
        empirical_means: Empirical means of the rewards of each arm
        confidence: Upper confidence interval dimension of each arm
    """

    def __init__(self, arms_values):
        """
        Initializes the UCB1 learner

        :param np.ndarray arms_values: Values associated to the arms
        """

        super().__init__(arms_values)
        self.successes_per_arm = [[] for _ in range(self.n_arms)]
        self.total_observations_per_arm = [[] for _ in range(self.n_arms)]
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)

    def pull_arm(self, cost, n_clicks, cum_daily_costs):
        """
        Chooses the arm to play based on the UCB1 algorithm, therefore choosing the arm with higher upper
        confidence bound, which is the mean of the reward of the arm plus the confidence interval

        :return: Index of the arm to pull
        :rtype: int
        """

        upper_confidence_bound = (self.empirical_means + self.confidence) * n_clicks * (self.arms_values - cost) - cum_daily_costs
        idx = np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])
        return idx

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updates the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))
        self.empirical_means[pulled_arm] = np.sum(self.successes_per_arm[pulled_arm]) / np.sum(self.total_observations_per_arm[pulled_arm])
        for arm in range(self.n_arms):
            self.confidence[arm] = np.sqrt(2 * np.log(self.t) / np.sum(self.total_observations_per_arm[arm])) if self.times_arms_played[arm] > 0 else np.inf


    def get_conv_prob(self, pulled_arm):
        """
        Returns the estimate conversion probability of the given arm

        :param int pulled_arm: Arm considered in the computation

        :returns: Estimate conversion probability of the given arm
        :rtype: float
        """

        # In else put 1 to be more optimistic in the case we don't have data
        return self.empirical_means[pulled_arm] if self.times_arms_played[pulled_arm] > 0 else 1

    def get_upper_confidence_bounds(self):
        """
        Returns the upper confidence bounds of all the arms

        :returns: Upper confidence bounds of all the arms
        :rtype: numpy.ndarray
        """

        return self.empirical_means + self.confidence

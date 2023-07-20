from Learner import *


class UCBLearner(Learner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) algorithm

    :param np.ndarray arms_values: Values associated to the arms
    empirical_means: Empirical means
    confidence: upper Confidence interval
    """

    def __init__(self, arms_values):
        super().__init__(arms_values)
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)

    def pull_arm(self):
        """
        Chooses the arm to play based on the UCB1 algorithm, therefore choosing the arm with higher upper
        confidence bound, which is the mean of the reward of the arm plus the confidence interval

        :return: Index of the arm to pull
        :rtype: int
        """

        upper_confidence_bound = self.empirical_means + self.confidence
        idx = np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

        return idx

    def update(self, pulled_arm, reward):
        """
        Updating the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.times_arms_played[pulled_arm] - 1) + reward) / self.times_arms_played[pulled_arm]
        for arm in range(self.n_arms):
            self.confidence[arm] = 100 * np.sqrt((2 * np.log(self.t) / self.times_arms_played[arm])) if self.times_arms_played[arm] > 0 else np.inf

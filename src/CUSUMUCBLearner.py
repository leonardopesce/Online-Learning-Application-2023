from UCB import UCBLearner
import numpy as np
from CUSUM import CUSUM


class CUSUMUCBLearner(UCBLearner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) with cumulative sum algorithm(CUSUM) to actively detect
    changes in non-stationary environments with abrupt changes

    Attributes:
        change_detection: List with a CUSUM for each arm
        valid_rewards_per_arm: List of rewards that are considered valid for each arm
        detections: List to store the times when a change is detected for each arm
    """

    def __init__(self, arms_values, M=15, eps=50, h=18, alpha=0.1):
        """
        Initializes the UCB1 learner with CUSUM

        :param np.ndarray arms_values: Values associated to the arms
        :param int M: M parameter for CUSUM
        :param float eps: Epsilon parameter for CUSUM
        :param float h: h parameter for CUSUM
        :param float alpha: Pure exploration parameter
        """

        super().__init__(arms_values)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(self.n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.detections = [[] for _ in range(self.n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        """
        Chooses the arm to play based on the UCB1 algorithm with probability 1-alpha, therefore choosing the arm with
        higher upper confidence bound, which is the mean of the reward of the arm plus the confidence interval. With
        probability alpha plays a random arm

        :return: Index of the arm to pull
        :rtype: int
        """

        return super().pull_arm() if np.random.binomial(1, 1 - self.alpha) else np.random.randint(0, self.n_arms)

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updates the attributes given the observations of the results obtained by playing the
        pulled arm in the environment checking if a change is detected

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        self.t += 1
        # Ask the change detector if flags a detection for the pulled arm
        if self.change_detection[pulled_arm].update(reward):
            # If flags a detection, restart the detection algorithm
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arm[pulled_arm] = []
            self.successes_per_arm[pulled_arm] = []
            self.total_observations_per_arm[pulled_arm] = []
            self.change_detection[pulled_arm].reset()

        self.update_observations(pulled_arm, reward)
        self.valid_rewards_per_arm[pulled_arm].append(reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arm[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arm])
        for arm in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arm[arm])
            self.confidence[arm] = 100 * np.sqrt((2 * np.log(total_valid_samples) / n_samples)) if n_samples > 0 else np.inf

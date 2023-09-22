import numpy as np

from .UCB import UCBLearner
from .CUSUM import CUSUM


class CUSUMUCBLearner(UCBLearner):
    """
    Learner that applies the Upper Confidence Bound 1(UCB1) with cumulative sum algorithm(CUSUM) to actively detect
    changes in non-stationary environments with abrupt changes

    Attributes:
        change_detection: List with a CUSUM for each arm
        valid_rewards_per_arm: List of rewards that are considered valid for each arm
        detections: List to store the times when a change is detected for each arm
        alpha: Pure exploration parameter
    """

    def __init__(self, arms_values, M, eps, h, alpha):
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

    def pull_arm(self, cost, n_clicks, cum_daily_costs):
        """
        Chooses the arm to play based on the UCB1 algorithm with probability 1-alpha, therefore choosing the arm with
        higher upper confidence bound, which is the mean of the reward of the arm plus the confidence interval. With
        probability alpha plays a random arm

        :param float cost: Other costs associated to all the prices
        :param np.ndarray n_clicks: Number of clicks
        :param np.ndarray cum_daily_costs: Cumulative daily costs

        :return: Index of the arm to pull
        :rtype: int
        """

        return super().pull_arm(cost, n_clicks, cum_daily_costs) if np.random.binomial(1, 1 - self.alpha) else np.random.randint(0, self.n_arms)

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
        for sample in bernoulli_realization:
            if self.change_detection[pulled_arm].update(sample):
                # If flags a detection, restart the detection algorithm
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arm[pulled_arm] = []
                self.successes_per_arm[pulled_arm] = []
                self.total_observations_per_arm[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
                #print("det")

        self.update_observations(pulled_arm, reward)
        self.valid_rewards_per_arm[pulled_arm].append(reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))
        self.empirical_means[pulled_arm] = np.sum(self.successes_per_arm[pulled_arm]) / np.sum(self.total_observations_per_arm[pulled_arm])

        total_valid_samples = np.sum([len(self.valid_rewards_per_arm[arm]) for arm in range(self.n_arms)])
        for arm in range(self.n_arms):
            self.confidence[arm] = np.sqrt((2 * np.log(total_valid_samples) / np.sum(self.total_observations_per_arm[arm]))) if len(self.valid_rewards_per_arm[arm]) > 0 else np.inf

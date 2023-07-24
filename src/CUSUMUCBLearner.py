from UCB import UCBLearner
import numpy as np
from CUSUM import CUSUM

# UCB for non-stationary environment,
# using change detection to deal with abrupt changes,
# that actively detects changes in the distribution of the rewards


class CUSUMUCBLearner(UCBLearner):
    def __init__(self, arms_values, M=5, eps=0.05, h=100, alpha=0.05): #TODO change this parameters, h = np.log(T)*2, h=50 funziona
        super().__init__(arms_values)
        # initialize a CUSUM for each arm
        self.change_detection = [CUSUM(M, eps, h) for _ in range(self.n_arms)]  # one CUSUM algorithm for each arm
        self.valid_rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.detections = [[] for _ in range(self.n_arms)]
        self.alpha = alpha  # pure exploration parameter

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            # with probability 1-alpha we select the arm that maximizes UCB
            return super().pull_arm()
        else:
            # play randomly
            return np.random.randint(0, self.n_arms)

    def update(self, pulled_arm, reward, bernoulli_realization):
        self.t += 1
        # for each pulled arm we ask the change detector for that arm if flags a detection
        if self.change_detection[pulled_arm].update(reward):
            # if flag a detection, initialize again valid rewards for that arm
            # and restart the detection algorithm
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

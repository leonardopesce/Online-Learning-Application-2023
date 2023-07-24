from UCB import UCBLearner
import numpy as np
from Cusum import CUSUM

class CUSUM_UCB_Learner(UCBLearner):
    def __init__(self, arms_values, M=100, eps=0.05, h=5, alpha=0.01):
        super().__init__(arms_values)
        # initialize a CUSUM for each arm
        self.change_detection = [CUSUM(M, eps, h) for _ in range(self.n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.detections = [[] for _ in range(self.n_arms)]
        # pure exploration parameter
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            # with probability 1-alpha we select the arm that maximizes UCB
            upper_conf = self.empirical_means + self.confidence
            idx = np.random.choice(np.where(upper_conf == upper_conf.max())[0])
            return idx
        else:
            random_idx = np.random.randint(0, self.n_arms)
            # solve optimization over a random matrix to get a random matching
            return random_idx

    def update(self, pulled_arm, reward, bernoulli_realization):
        self.t += 1
        # for each pulled arm we ask the change detector for that arm if flags a detection
        if self.change_detection[pulled_arm].update(reward):
            # if flag a detection, initialize again valid rewards for that arm
            # and restart the detection algorithm
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arm[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arm[pulled_arm])
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arm])
        for arm in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arm[arm])
            self.confidence[arm] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arm[pulled_arm].append(reward)
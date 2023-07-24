from UCB import *


# Sliding Window UCB
class SWUCBLearner(UCBLearner):

    def __init__(self, arms_values, window_size):
        super().__init__(arms_values)
        self.window_size = window_size

    def update(self, pulled_arm, reward, bernoulli_realization):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.successes_per_arm[pulled_arm] += np.sum(bernoulli_realization)
        self.total_observations_per_arm[pulled_arm] += len(bernoulli_realization)
        # For each arm slide the window over each we compute the estimate
        for arm in range(self.n_arms):
            # Count the number of times we pulled the current arm in the window
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)

            self.empirical_means[pulled_arm] = np.sum(self.rewards_per_arm[arm][-n_samples:]) / n_samples if n_samples > 0 else 0
            self.confidence[arm] = 1000 * np.sqrt((2 * np.log(self.t) / n_samples)) if n_samples > 0 else np.inf

    # check if an arm was never played during the window, if yes play it, this should be automatically done
    # because in the update of UCB if n_samples is 0 the confidence goes to infinity and so the arm will be pulled
    # the next round


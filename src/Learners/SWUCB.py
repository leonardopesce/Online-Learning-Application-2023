from .UCB import *


class SWUCBLearner(UCBLearner):
    """
    Learner that applies the UCB1 algorithm with sliding window(SW) to passively detect changes in non-stationary
    environments
    """

    def __init__(self, arms_values, window_size):
        """
        Initializes the UCB1 learner with SW

        :param np.ndarray arms_values: Values associated to the arms
        :param int window_size: Size of the sliding window
        """

        super().__init__(arms_values)
        self.window_size = window_size

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updates the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        # Check if an arm was never played during the window, if yes play it. This should be automatically done
        # because in the update of UCB if n_samples is 0 the confidence goes to infinity and so the arm will be pulled
        # the next round
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))
        # For each arm slide the window over which we compute the estimate
        #total_valid_samples = 0
        #for arm in range(self.n_arms):
         #   n_samples = np.sum(np.array(self.pulled_arms[-self.window_size:]) == arm)
         #   total_valid_samples += np.sum(self.total_observations_per_arm[arm][-n_samples:])
        for arm in range(self.n_arms):
            # Count the number of times we pulled the current arm in the window
            n_samples = np.sum(np.array(self.pulled_arms[-self.window_size:]) == arm)
            self.empirical_means[arm] = np.sum(self.successes_per_arm[arm][-n_samples:]) / np.sum(self.total_observations_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            #self.confidence[arm] = np.sqrt((2 * np.log(total_valid_samples) / np.sum(self.total_observations_per_arm[arm][-n_samples:]))) if n_samples > 0 else np.inf
            self.confidence[arm] = np.sqrt((2 * np.log(self.t) / np.sum(self.total_observations_per_arm[arm][-n_samples:]))) if n_samples > 0 else np.inf

    def get_conv_prob(self, pulled_arm):
        """
        Returns the estimate conversion probability of the given arm

        :param int pulled_arm: Arm considered in the computation

        :returns: Estimate conversion probability of the given arm
        :rtype: float
        """

        n_samples = np.sum(np.array(self.pulled_arms[-self.window_size:]) == pulled_arm)
        return self.empirical_means[pulled_arm] if n_samples > 0 else 1


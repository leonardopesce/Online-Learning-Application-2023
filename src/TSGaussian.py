from Learner import *


class TSGaussianLearner(Learner):
    """
    Learner that applies the Gaussian Thompson Sampling(TS) algorithm

    :param int n_arms: Number of arms
    means: Means of the arms
    sigmas: Standard devitions of the arms
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initialize the parameters for the Gaussian distribution (mean and variance) for each arm
        self.means = np.zeros(self.n_arms)
        #self.sigmas = np.ones(self.n_arms) * 1e2
        self.variances = np.ones(self.n_arms) * 10

    def pull_arm(self):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the gaussian distribution and
        choosing the arm from whose distribution is extracted the maximum value

        :return: Index of the arm to pull
        :rtype: int
        """

        idx = np.argmax(np.random.normal(self.means, np.sqrt(self.variances)))
        return idx

    def update(self, pulled_arm, reward):
        """
        Updating mean and variance of the gaussian distribution given the observations of the results obtained by playing
        the pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        # TODO check the formulas should be the empirical mean and std
        self.t += 1
        self.update_observations(pulled_arm, reward)
        #self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        #n_samples = len(self.rewards_per_arm[pulled_arm])
        #if n_samples > 1:
        #   self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples
        self.means[pulled_arm] = (self.means[pulled_arm] * (self.times_arms_played[pulled_arm] - 1) + reward) / self.times_arms_played[pulled_arm]
        if self.times_arms_played[pulled_arm] > 1:
            self.variances[pulled_arm] = ((self.variances[pulled_arm] * (self.times_arms_played[pulled_arm] - 1)) + (reward - self.means[pulled_arm]) ** 2) / self.times_arms_played[pulled_arm]

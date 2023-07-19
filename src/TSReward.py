from Learner import *


class TSRewardLearner(Learner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm

    :param int n_arms: Number of arms
    beta_parameters: Parameters of the Beta distribution, one for each arm
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initializing the Beta distribution of each arm to a uniform distribution
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self, arms_values, cost):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the Beta distribution and
        choosing the arm from whose distribution is extracted the maximum value

        :return: Index of the arm to pull
        :rtype: int
        """

        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * (arms_values - cost))
        return idx

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updating alpha and beta of the beta distribution given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + bernoulli_realization
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - bernoulli_realization

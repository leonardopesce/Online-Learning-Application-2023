import numpy as np
from Learner import *


class TSLearner(Learner):
    """
    Learner that applies the Thompson Sampling algorithm.
    beta_parameters: parameters of the Beta distribution, one for each arm.
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initializing the Beta distribution of each arm to a uniform distribution
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        """
        Chooses the arm to play based on the Thompson Sampling algorithm, therefore sampling the Beta distribution and
        choosing the arm from whose distribution is extracted the maximum value.

        :return: index of the arm to pull.
        """

        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        """
        Updating alpha and beta of the beta distribution given the observations of the results obtained by playing the
        pulled arm in the environment.

        :param pulled_arm: arm pulled in the current time step.
        :param reward: reward collected in the current time step playing the pulled arm.
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

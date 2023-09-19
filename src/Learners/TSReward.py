from .Learner import *


class TSRewardLearner(Learner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm taking into account the reward is not binary

    Attributes:
        beta_parameters: Parameters of the Beta distribution, one for each arm
    """

    def __init__(self, arms_values):
        """
        Initializes the TS learner

        :param numpy.ndarray arms_values: Values associated to the arms
        """
        super().__init__(arms_values)
        # Initializing the Beta distribution of each arm to a uniform distribution
        self.beta_parameters = np.ones((self.n_arms, 2))

    def pull_arm(self, cost, n_clicks, cum_daily_costs):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the Beta distribution and
        choosing the arm from whose distribution is extracted the maximum value multiplied by the margin (price - cost)

        :param float cost: Cost associated to all the arms
        :param numpy.ndarray n_clicks: Number of clicks
        :param numpy.ndarray cum_daily_costs: Cumulative daily costs

        :return: Index of the arm to pull
        :rtype: int
        """

        idx = np.argmax(n_clicks * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * (self.arms_values - cost) - cum_daily_costs)
        return idx

    def update(self, pulled_arm, reward, bernoulli_realizations):
        """
        Updates alpha and beta of the beta distributions given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param numpy.ndarray bernoulli_realizations: Bernoulli realization of the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + np.sum(bernoulli_realizations)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + np.sum(1 - bernoulli_realizations)

    def get_conv_prob(self, pulled_arm):
        """
        Returns the estimate conversion probability of the given arm

        :param int pulled_arm: Arm considered in the computation

        :returns: Estimate conversion probability of the given arm
        :rtype: float
        """

        return self.beta_parameters[pulled_arm, 0] / (self.beta_parameters[pulled_arm, 0] + self.beta_parameters[pulled_arm, 1])

    def get_betas(self):
        """
        Returns the parameters of the beta distributions od all the arms

        :returns: Parameters of the beta distributions od all the arms
        :rtype: numpy.ndarray
        """

        return self.beta_parameters

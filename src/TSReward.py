from Learner import *


class TSRewardLearner(Learner):
    """
    Learner that applies the Thompson Sampling(TS) algorithm taking into account the reward is not binary

    :param np.ndarray arms_values: Values associated to the arms
    beta_parameters: Parameters of the Beta distribution, one for each arm
    """

    def __init__(self, arms_values):
        super().__init__(arms_values)
        # Initializing the Beta distribution of each arm to a uniform distribution
        self.beta_parameters = np.ones((self.n_arms, 2))

    def pull_arm(self, arms_values, cost, n_clicks, cum_daily_costs):
        """
        Chooses the arm to play based on the TS algorithm, therefore sampling the Beta distribution and
        choosing the arm from whose distribution is extracted the maximum value multiplied by the margin (price - cost)

        :param np.ndarray arms_values: Array with the prices associated to the arms
        :param float cost: Cost associated to all the arms
        :param np.ndarray n_clicks: Number of clicks
        :param np.ndarray cum_daily_costs: Cumulative daily costs
        :return: Index of the arm to pull
        :rtype: int
        """
        #TODO che sia da considerare anche la parte del'adv? al momento la consideriamo
        idx = np.argmax(n_clicks * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * (arms_values - cost) - cum_daily_costs)
        return idx

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updating alpha and beta of the beta distribution given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + np.sum(bernoulli_realization)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + np.sum(1 - bernoulli_realization)

    def get_conv_prob(self, pulled_arm):
        return self.beta_parameters[pulled_arm, 0] / (self.beta_parameters[pulled_arm, 0] + self.beta_parameters[pulled_arm, 1])

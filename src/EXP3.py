from Learner import *


class EXP3Learner(Learner):
    """
    Learner that applies the Exponential-weight algorithm for Exploration and Exploitation(EXP3) algorithm

    Attributes:
        successes_per_arm: Number of times each arm has obtained a positive realization
        total_observations_per_arm: Total number of realizations of each arm
        weights: List with the weight for each arm
        p: List with the probability for each arm
    """

    def __init__(self, arms_values, worst_reward, best_reward, other_costs, gamma=0.5):
        """
        Initializes the EXP3 learner

        :param np.ndarray arms_values: Values associated to the arms
        :param float worst_reward: Worst possible reward
        :param float best_reward: Best possible reward
        :param float gamma: Exploration parameter
        :param float other_costs: Cost of each arm
        """

        super().__init__(arms_values)
        self.successes_per_arm = [[] for _ in range(self.n_arms)]
        self.total_observations_per_arm = [[] for _ in range(self.n_arms)]

        self.worst_reward = worst_reward
        self.best_reward = best_reward
        self.gamma = gamma
        self.weights = np.ones(self.n_arms)
        self.p = np.ones(self.n_arms) / self.n_arms
        self.other_costs = other_costs

    def pull_arm(self):
        """
        Chooses the arm to play based on the EXP3 algorithm, therefore sampling according to the probabilities of each arm

        :return: Index of the arm to pull
        :rtype: int
        """

        self.p = [(1 - self.gamma) * weight / np.sum(self.weights) + self.gamma / self.n_arms for weight in self.weights]
        idx = np.random.choice(self.n_arms, size=1, p=self.p)[0]

        return idx

    def update(self, pulled_arm, reward, bernoulli_realization):
        """
        Updates the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))

        for realization in bernoulli_realization:
            reward = (realization * (self.arms_values[pulled_arm] - self.other_costs) - self.worst_reward) / (self.best_reward - self.worst_reward)
            self.weights[pulled_arm] = self.weights[pulled_arm] * np.exp(self.gamma * reward / (self.p[pulled_arm] * self.n_arms))

    def update1(self, pulled_arm, reward, bernoulli_realization):
        """
        Updates the attributes given the observations of the results obtained by playing the
        pulled arm in the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        :param bernoulli_realization: Bernoulli realization of the pulled arm
        """

        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.successes_per_arm[pulled_arm].append(np.sum(bernoulli_realization))
        self.total_observations_per_arm[pulled_arm].append(len(bernoulli_realization))

        reward = (reward - self.worse_reward) / (self.best_reward - self.worse_reward)
        self.weights[pulled_arm] = self.weights[pulled_arm] * np.exp(self.gamma * reward / (self.p[pulled_arm] * self.n_arms))

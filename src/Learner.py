import numpy as np


class Learner:
    """
    Superclass of all the types of learner

    Attributes:
        t: Current time step
        pulled_arms: Arms played in each time step
        rewards_per_arm: Rewards obtained for each arm
        collected_rewards: Rewards collected in each time step
        times_arms_played: Number of times that an arm has been played
    """

    def __init__(self, arms_values):
        """
        Initializes the learner

        :param np.ndarray arms_values: Values associated to the arms
        """

        self.n_arms = len(arms_values)
        self.arms_values = arms_values
        self.t = 0
        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.times_arms_played = np.zeros(self.n_arms)

    def pull_arm(self):
        """
        Pulls the arm to play, in this "abstract" learner class, it does nothing
        """
        return

    def update_observations(self, pulled_arm, reward):
        """
        Updates the observations lists, once the reward is returned by the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param float reward: Reward collected in the current time step playing the pulled arm
        """

        self.pulled_arms.append(pulled_arm)
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.times_arms_played[pulled_arm] += 1

    def get_arms(self):
        """
        Returns the values of the arms that the learner can play

        :return: Array of the arms of the learner
        :rtype: numpy.ndarray
        """

        return self.arms_values

import numpy as np


class Learner:
    """
    Superclass of all the types of learner

    :param int n_arms: Number of arms the learner can play
    t: Current time step
    pulled_arms: Arms played in each time step
    rewards_per_arm: Rewards obtained for each arm
    collected_rewards: Rewards collected in each time step
    times_arms_played: Number of times that an arm has been played
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.times_arms_played = np.zeros(n_arms)

    def pull_arm(self):
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

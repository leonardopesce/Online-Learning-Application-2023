import numpy as np


class Learner:
    """
    Superclass of all the types of learner.
    # TO DO specify what are the parameters (formato da decidere)
    n_arms: number of arms the learner can play.
    t: current round step.
    pulled_arms: arms played in each time step.
    rewards_per_arm: rewards obtained divided by arm.
    collected_rewards: rewards collected in each time step.

    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def pull_arm(self):
        return

    def update_observations(self, pulled_arm, reward):
        """
        Updates the observations lists, once the reward is returned by the environment.

        :param pulled_arm: arm pulled in the current time step;
        :param reward: reward collected in the current time step playing the pulled arm.
        """

        self.pulled_arms.append(pulled_arm)
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

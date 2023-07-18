import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from UCB import *
from TS import *


# Considered category is C1
category = 'C1'

# Setting of the environment parameters
n_arms = 5
arms_values = {'C1': np.array([500, 550, 600, 650, 700]),
               'C2': np.array([500, 550, 600, 650, 700]),
               'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([1, 1, 0.5]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([100, 0.5, 0.5]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 200

# TODO need to create the clairvoyant
# last one is the optimal one
#opt = p[3]

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform at least 1000 experiments
n_experiments = 1000

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# Each iteration simulates the learner-environment interaction
for e in range(0, n_experiments):
    # Define the environment
    env = Environment(n_arms, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
    # Define the learners
    ts_learner = TSLearner(n_arms)
    ucb_learner = UCBLearner(n_arms)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        pulled_arm = ts_learner.pull_arm()
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        reward = env.reward(category, pulled_arm, bernoulli_realization)
        ts_learner.update(pulled_arm, reward)

        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        reward = env.round_pricing(category, pulled_arm, bernoulli_realization)
        ucb_learner.update(pulled_arm, reward)

    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

# Plot the results
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Reward")

plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')  # red for TS
plt.plot(np.cumsum(np.mean(ucb_reward_per_experiment, axis=0)), 'g')  # green for UCB
plt.legend(["TS", "UCB"])
plt.show()

# plot the results
#plt.figure(0)
#plt.xlabel("t")
#plt.ylabel("Regret")
# we plot the Regret of the algorithms
# which is the cumulative sum of the difference from the optimum and the reward collected by the agent
# we compute the mean over all the experiments
#plt.plot(np.cumsum(np.mean(opt - ts_reward_per_experiment, axis=0)), 'r')  # red for TS
#plt.plot(np.cumsum(np.mean(opt - gr_reward_per_experiment, axis=0)), 'g')  # green for Greedy
#plt.legend(["TS", "Greedy"])
#plt.show()

# the regret of the greedy algorithm increases linearly,
# while the instantaneous regret of TS decreases as the number of round increases

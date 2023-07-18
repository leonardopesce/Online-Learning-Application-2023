import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from UCB import *
from TS import *
# Considered category is C1
category = 'C1'

# create an experiment to compare performance
# define basic setting with 5 arms
n_arms = 5

# Bernoulli distribution for the reward functions of the arms
#p = np.array([0.15, 0.1, 0.1, 0.35])

# last one is the optimal one
#opt = p[3]

# horizon of the experiment
T = 365

# since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform at least 1000 experiments
n_experiments = 1000

# store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# in each iteration we simulate the learner-environment interaction
for e in range(0, n_experiments):
    # define environment with the arms and prob distr defined before
    env = Environment()
    # define the learners
    ts_learner = TSLearner(n_arms)
    ucb_learner = UCBLearner(n_arms)

    # iterate on the number of rounds
    for t in range(0, T):
        # simulate the interaction learner-environment
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        reward = env.reward(category, pulled_arm, bernoulli_realization)
        ts_learner.update(pulled_arm, reward)

        # Upper Confidence Bound Learner
        pulled_arm = ucb_learner.pull_arm()
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        reward = env.round_pricing(category, pulled_arm, bernoulli_realization)
        ucb_learner.update(pulled_arm, reward)

    # store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

# plot the results
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

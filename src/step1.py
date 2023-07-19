from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from TS import *
from TSReward import *

"""
Simulation for the step 1: learning for pricing 
Consider the case in which all the users belong to class C1. Assume that the curves related to the advertising part of 
the problem are known, while the curve related to the pricing problem is not. Apply the UCB1 and TS algorithms, 
reporting the plots of the average (over a sufficiently large number of runs) value and standard deviation of the 
cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

# Setting the environment parameters
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
bids_to_cum_costs = {'C1': np.array([10, 0.5, 0.5]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 300

# Time horizon of the experiment
T = 600

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform at least 300 experiments
n_experiments = 1000

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# TODO maybe move out the env and clairvoyant
# Define the environment
env = Environment(n_arms, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)

ucb_best = []
ts_best = []

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    ts_learner = TSRewardLearner(n_arms)
    ucb_learner = UCBLearner(n_arms)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        pulled_arm = ts_learner.pull_arm(arms_values[category], other_costs)
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        #conversion_times_margin = env.get_conversion_times_margin(category, pulled_arm, conversion_probability=bernoulli_realization)
        #_, _, reward = clairvoyant.maximize_reward_from_bid(category, conversion_times_margin)
        reward = env.get_reward_from_price(category, pulled_arm, bernoulli_realization, best_bid_idx)
        ts_learner.update(pulled_arm, reward, bernoulli_realization)

        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        bernoulli_realization = env.round_pricing(pulled_arm, category)
        #conversion_times_margin = env.get_conversion_times_margin(category, pulled_arm, conversion_probability=bernoulli_realization)
        #_, _, reward = clairvoyant.maximize_reward_from_bid(category, conversion_times_margin)
        reward = env.get_reward_from_price(category, pulled_arm, bernoulli_realization, best_bid_idx)
        ucb_learner.update(pulled_arm, reward)
        #if t % 100 == 0 and e % 10 == 0:
           # print("============================")
           # print(ucb_learner.confidence+ucb_learner.empirical_means)

    ucb_best.append(np.argmax(ucb_learner.empirical_means + ucb_learner.confidence))
    ts_best.append(np.argmax(ts_learner.beta_parameters[:, 0] / (ts_learner.beta_parameters[:, 0] + ts_learner.beta_parameters[:, 1])))
    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

# Plot the results
_, axes = plt.subplots(1, 2)
#axes[0].xlabel("t")
#axes[0].ylabel("Reward")
axes[0].set_title('Regret')
axes[0].plot(np.cumsum(np.mean(best_reward - ts_reward_per_experiment, axis=0)), 'r')
axes[0].plot(np.cumsum(np.mean(best_reward - ucb_reward_per_experiment, axis=0)), 'g')
#plt.fill_between(np.arange(0, T, 1), np.cumsum(np.mean(ts_reward_per_experiment, axis=0)) - (np.std(ts_reward_per_experiment, axis=0)), np.cumsum(np.mean(ts_reward_per_experiment, axis=0)) + (np.std(ts_reward_per_experiment, axis=0)), color='r', alpha=0.4)
axes[0].legend(["TS", "UCB"])

axes[1].set_title('Reward')
best_rewards = np.ones((T,)) * best_reward
axes[1].plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
axes[1].plot(np.cumsum(np.mean(ucb_reward_per_experiment, axis=0)), 'g')
axes[1].plot(np.cumsum(best_rewards), 'b')
axes[1].legend(["TS", "UCB", "clairvoyant"])
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

from collections import Counter



occurrences_ucb = Counter(ucb_best)
print(occurrences_ucb)

occurrences_ts = Counter(ts_best)
print(occurrences_ts)

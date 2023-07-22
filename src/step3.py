from matplotlib import pyplot as plt

from Environment import Environment
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising
from collections import Counter
import numpy as np

"""
Simulation for the step 3: Learning for joint pricing and advertising
Consider the case in which all the users belong to class C1, and no information about the advertising and pricing curves
is known beforehand. Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves, reporting
the plots of the average (over a sufficiently large number of runs) value and standard deviation of the cumulative 
regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.03, 0.04, 0.05, 0.03, 0.01]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([100, 2, 0.0]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([20, 0.5, 0.0]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 400

# Bids setup
n_bids = 100
min_bid = 0.5
max_bid = 20.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma = 2

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 50



# To evaluate which are the most played prices and bids
ts_best_price = []
ts_best_bid = []
ucb_best_price = []
ucb_best_bid = []

# Define the environment
env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# Store some features for each experiment for the learners
gpts_clicks_per_experiment = []
gpts_mean_clicks_per_experiment = []
gpts_sigmas_clicks_per_experiment = []
gpts_cum_costs_per_experiment = []
gpts_mean_cum_costs_per_experiment = []
gpts_sigmas_cum_costs_per_experiment = []
gpts_pulled_bids_per_experiment = []

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    # TS learner
    ts_learner = TSLearnerPricingAdvertising(env.prices[category], env.bids)
    # UCB learner
    ucb_learner = UCBLearnerPricingAdvertising(env.prices[category], env.bids)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        price_idx, bid_idx = ts_learner.pull_arm(env.other_costs)
        bernoulli_realizations, n_clicks, cum_daily_cost = env.round(category, price_idx, bid_idx)
        reward = env.get_reward(category, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
        ts_learner.update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

        # UCB Learner
        price_idx, bid_idx = ucb_learner.pull_arm(env.other_costs)
        bernoulli_realizations, n_clicks, cum_daily_cost = env.round(category, price_idx, bid_idx)
        reward = env.get_reward(category, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
        ucb_learner.update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

    # Store the most played prices and bids by TS
    ts_best_price.append(Counter(ts_learner.get_pulled_prices()).most_common(1)[0][0])
    ts_best_bid.append(Counter(ts_learner.get_pulled_bids()).most_common(1)[0][0])

    # Store the most played prices and bids by UCB1
    ucb_best_price.append(Counter(ucb_learner.get_pulled_prices()).most_common(1)[0][0])
    ucb_best_bid.append(Counter(ucb_learner.get_pulled_bids()).most_common(1)[0][0])

    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.TS_pricing.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.UCB_pricing.collected_rewards)

    gpts_clicks_per_experiment.append(ts_learner.GPTS_advertising.collected_clicks)
    gpts_mean_clicks_per_experiment.append(ts_learner.GPTS_advertising.means_clicks)
    gpts_sigmas_clicks_per_experiment.append(ts_learner.GPTS_advertising.sigmas_clicks)
    gpts_cum_costs_per_experiment.append(ts_learner.GPTS_advertising.collected_costs)
    gpts_mean_cum_costs_per_experiment.append(ts_learner.GPTS_advertising.means_costs)
    gpts_sigmas_cum_costs_per_experiment.append(ts_learner.GPTS_advertising.sigmas_costs)
    gpts_pulled_bids_per_experiment.append(ts_learner.GPTS_advertising.pulled_bids)

# Print occurrences of best arm in TS
print(Counter(ts_best_price))
print(Counter(ts_best_bid))
# Print occurrences of best arm in UCB1
print(Counter(ucb_best_price))
print(Counter(ucb_best_bid))

# Plot the results, comparison TS-UCB
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
best_rewards = np.ones((T,)) * best_reward
regret_ts_mean = np.mean(best_reward - ts_reward_per_experiment, axis=0)
regret_ts_std = np.std(best_reward - ts_reward_per_experiment, axis=0)
regret_ucb_mean = np.mean(best_reward - ucb_reward_per_experiment, axis=0)
regret_ucb_std = np.std(best_reward - ucb_reward_per_experiment, axis=0)
cumulative_regret_ts_mean = np.mean(np.cumsum(best_reward - ts_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ts_std = np.std(np.cumsum(best_reward - ts_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ucb_mean = np.mean(np.cumsum(best_reward - ucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ucb_std = np.std(np.cumsum(best_reward - ucb_reward_per_experiment, axis=1), axis=0)
reward_ts_mean = np.mean(ts_reward_per_experiment, axis=0)
reward_ts_std = np.std(ts_reward_per_experiment, axis=0)
reward_ucb_mean = np.mean(ucb_reward_per_experiment, axis=0)
reward_ucb_std = np.std(ucb_reward_per_experiment, axis=0)
cumulative_reward_ts_mean = np.mean(np.cumsum(ts_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ts_std = np.std(np.cumsum(ts_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ucb_mean = np.mean(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ucb_std = np.std(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)

axes[0].set_title('Instantaneous regret plot')
axes[0].plot(regret_ts_mean, 'r')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS", "UCB"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot')
axes[1].plot(reward_ts_mean, 'r')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS", "UCB", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].legend(["TS", "UCB"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot')
axes[3].plot(cumulative_reward_ts_mean, 'r')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["TS", "UCB", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# Plot the results for TS with std
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

axes[0].set_title('Instantaneous regret plot for TS')
axes[0].plot(regret_ts_mean, 'r')
axes[0].fill_between(range(0, T), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS mean", "TS std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot for TS')
axes[1].plot(reward_ts_mean, 'r')
axes[1].fill_between(range(0, T), reward_ts_mean - reward_ts_std, reward_ts_mean + reward_ts_std, color='r', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS mean", "TS std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot for TS')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].fill_between(range(0, T), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='r', alpha=0.4)
axes[2].legend(["TS mean", "TS std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot for TS')
axes[3].plot(cumulative_reward_ts_mean, 'r')
axes[3].fill_between(range(0, T), cumulative_reward_ts_mean - cumulative_reward_ts_std, cumulative_reward_ts_mean + cumulative_reward_ts_std, color='r', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["TS mean", "TS std", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# Plot the results for UCB with std
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

axes[0].set_title('Instantaneous regret plot for UCB')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].fill_between(range(0, T), regret_ucb_mean - regret_ucb_std, regret_ucb_mean + regret_ucb_std, color='g', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["UCB mean", "UCB std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot for UCB')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].fill_between(range(0, T), reward_ucb_mean - reward_ucb_std, reward_ucb_mean + reward_ucb_std, color='g', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot for UCB')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].fill_between(range(0, T), cumulative_regret_ucb_mean - cumulative_regret_ucb_std, cumulative_regret_ucb_mean + cumulative_regret_ucb_std, color='g', alpha=0.4)
axes[2].legend(["UCB mean", "UCB std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot for UCB')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].fill_between(range(0, T), cumulative_reward_ucb_mean - cumulative_reward_ucb_std, cumulative_reward_ucb_mean + cumulative_reward_ucb_std, color='g', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")

plt.show()

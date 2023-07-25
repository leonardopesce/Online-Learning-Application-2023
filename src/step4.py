from matplotlib import pyplot as plt

from Environment import Environment
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising
from collections import Counter
import numpy as np

"""
Consider the case in which there are three classes of users (C1, C2, and C3), 
and no information about the advertising and pricing curves is known beforehand. 

Consider two scenarios. 
In the first one, the structure of the contexts is known beforehand. 
Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves, 
reporting the plots with the average (over a sufficiently large number of runs) value and standard deviation 
of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward. 

In the second scenario, the structure of the contexts is not known beforehand and needs to be learnt from data. 
Important remark: the learner does not know how many contexts there are, 
while it can only observe the features and data associated with the features. 
Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves paired with 
a context generation algorithm, reporting the plots with the average (over a sufficiently large number of runs) 
value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, 
and instantaneous reward. Apply the context generation algorithms every two weeks of the simulation. 
Compare the performance of the two algorithms --- the one used in the first scenario with 
the one used in the second scenario. Furthermore, in the second scenario, 
run the GP-UCB and GP-TS algorithms without context generation, and therefore forcing the context to be only one 
for the entire time horizon, and compare their performance with the performance of the previous algorithms used 
for the second scenario.
"""

# Considered category is C1
category = 'C1'
classes = ['C1', 'C2', 'C3']

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.03, 0.04, 0.05, 0.03, 0.01]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([100, 2]),
                  'C2': np.array([2, 2]),
                  'C3': np.array([3, 3])}
bids_to_cum_costs = {'C1': np.array([20, 0.5]),
                     'C2': np.array([2, 2]),
                     'C3': np.array([3, 3])}
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
n_experiments = 20



# To evaluate which are the most played prices and bids
ts_best_price = {cl: [] for cl in classes}
ts_best_bid = {cl: [] for cl in classes}
ucb_best_price = {cl: [] for cl in classes}
ucb_best_bid = {cl: [] for cl in classes}


# Define the environment
env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = {cl: Clairvoyant(env) for cl in classes}
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = {cl: [] for cl in classes}, {cl: [] for cl in classes}, {cl: [] for cl in classes}, {cl: [] for cl in classes}, {cl: [] for cl in classes}
for cl in classes:
    best_price_idx[cl], best_price[cl], best_bid_idx[cl], best_bid[cl], best_reward[cl] = clairvoyant[cl].maximize_reward(cl)

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = {cl: [] for cl in classes}
ucb_reward_per_experiment = {cl: [] for cl in classes}

# Store some features for each experiment for the learners
gpts_clicks_per_experiment = {cl: [] for cl in classes}
gpts_mean_clicks_per_experiment = {cl: [] for cl in classes}
gpts_sigmas_clicks_per_experiment = {cl: [] for cl in classes}
gpts_cum_costs_per_experiment = {cl: [] for cl in classes}
gpts_mean_cum_costs_per_experiment = {cl: [] for cl in classes}
gpts_sigmas_cum_costs_per_experiment = {cl: [] for cl in classes}
gpts_pulled_bids_per_experiment = {cl: [] for cl in classes}

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    # TS learners
    ts_learner = {cl: TSLearnerPricingAdvertising(env.prices[cl], env.bids) for cl in classes}
    # UCB learners
    ucb_learner = {cl: UCBLearnerPricingAdvertising(env.prices[cl], env.bids) for cl in classes}

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        for cl in classes:
            # TS Learner
            price_idx, bid_idx = ts_learner[cl].pull_arm(env.other_costs)
            # this may simulate that all the users are of the same class
            bernoulli_realizations, n_clicks, cum_daily_cost = env.round(cl, price_idx, bid_idx)
            reward = env.get_reward(cl, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
            ts_learner[cl].update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

            # UCB Learner
            price_idx, bid_idx = ucb_learner[cl].pull_arm(env.other_costs)
            bernoulli_realizations, n_clicks, cum_daily_cost = env.round(cl, price_idx, bid_idx)
            reward = env.get_reward(cl, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
            ucb_learner[cl].update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

    for cl in classes:
        # Store the most played prices and bids by TS
        ts_best_price[cl].append(Counter(ts_learner[cl].get_pulled_prices()).most_common(1)[0][0])
        ts_best_bid[cl].append(Counter(ts_learner[cl].get_pulled_bids()).most_common(1)[0][0])

        # Store the most played prices and bids by UCB1
        ucb_best_price[cl].append(Counter(ucb_learner[cl].get_pulled_prices()).most_common(1)[0][0])
        ucb_best_bid[cl].append(Counter(ucb_learner[cl].get_pulled_bids()).most_common(1)[0][0])

        # Store the values of the collected rewards of the learners
        ts_reward_per_experiment[cl].append(ts_learner[cl].TS_pricing.collected_rewards)
        ucb_reward_per_experiment[cl].append(ucb_learner[cl].UCB_pricing.collected_rewards)

        gpts_clicks_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.collected_clicks)
        gpts_mean_clicks_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.means_clicks)
        gpts_sigmas_clicks_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.sigmas_clicks)
        gpts_cum_costs_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.collected_costs)
        gpts_mean_cum_costs_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.means_costs)
        gpts_sigmas_cum_costs_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.sigmas_costs)
        gpts_pulled_bids_per_experiment[cl].append(ts_learner[cl].GPTS_advertising.pulled_bids)

for cl in classes:
    print(f'Class {cl}:')
    # Print occurrences of best arm in TS
    print(Counter(ts_best_price[cl]))
    print(Counter(ts_best_bid[cl]))
    # Print occurrences of best arm in UCB1
    print(Counter(ucb_best_price[cl]))
    print(Counter(ucb_best_bid[cl]))

#plot data
best_rewards = {cl: [] for cl in classes}
regret_ts_mean = {cl: [] for cl in classes}
regret_ts_std = {cl: [] for cl in classes}
regret_ucb_mean = {cl: [] for cl in classes}
regret_ucb_std = {cl: [] for cl in classes}
cumulative_regret_ts_mean = {cl: [] for cl in classes}
cumulative_regret_ts_std = {cl: [] for cl in classes}
cumulative_regret_ucb_mean = {cl: [] for cl in classes}
cumulative_regret_ucb_std = {cl: [] for cl in classes}
reward_ts_mean = {cl: [] for cl in classes}
reward_ts_std = {cl: [] for cl in classes}
reward_ucb_mean = {cl: [] for cl in classes}
reward_ucb_std = {cl: [] for cl in classes}
cumulative_reward_ts_mean = {cl: [] for cl in classes}
cumulative_reward_ts_std = {cl: [] for cl in classes}
cumulative_reward_ucb_mean = {cl: [] for cl in classes}
cumulative_reward_ucb_std = {cl: [] for cl in classes}
for cl in classes:
    best_rewards[cl] = np.ones((T,)) * best_reward[cl]
    regret_ts_mean[cl] = np.mean(best_reward[cl] - ts_reward_per_experiment[cl], axis=0)
    regret_ts_std[cl] = np.std(best_reward[cl] - ts_reward_per_experiment[cl], axis=0)
    regret_ucb_mean[cl] = np.mean(best_reward[cl] - ucb_reward_per_experiment[cl], axis=0)
    regret_ucb_std[cl] = np.std(best_reward[cl] - ucb_reward_per_experiment[cl], axis=0)
    cumulative_regret_ts_mean[cl] = np.mean(np.cumsum(best_reward[cl] - ts_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_regret_ts_std[cl] = np.std(np.cumsum(best_reward[cl] - ts_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_regret_ucb_mean[cl] = np.mean(np.cumsum(best_reward[cl] - ucb_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_regret_ucb_std[cl] = np.std(np.cumsum(best_reward[cl] - ucb_reward_per_experiment[cl], axis=1), axis=0)
    reward_ts_mean[cl] = np.mean(ts_reward_per_experiment[cl], axis=0)
    reward_ts_std[cl] = np.std(ts_reward_per_experiment[cl], axis=0)
    reward_ucb_mean[cl] = np.mean(ucb_reward_per_experiment[cl], axis=0)
    reward_ucb_std[cl] = np.std(ucb_reward_per_experiment[cl], axis=0)
    cumulative_reward_ts_mean[cl] = np.mean(np.cumsum(ts_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_reward_ts_std[cl] = np.std(np.cumsum(ts_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_reward_ucb_mean[cl] = np.mean(np.cumsum(ucb_reward_per_experiment[cl], axis=1), axis=0)
    cumulative_reward_ucb_std[cl] = np.std(np.cumsum(ucb_reward_per_experiment[cl], axis=1), axis=0)

# Plot the results, comparison TS-UCB
for cl in classes:
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    axes[0].set_title(f'Instantaneous regret plot {cl}')
    axes[0].plot(regret_ts_mean[cl], 'r')
    axes[0].plot(regret_ucb_mean[cl], 'g')
    axes[0].axhline(y=0, color='b', linestyle='--')
    axes[0].legend(["TS", "UCB"])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Instantaneous regret")

    axes[1].set_title(f'Instantaneous reward plot {cl}')
    axes[1].plot(reward_ts_mean[cl], 'r')
    axes[1].plot(reward_ucb_mean[cl], 'g')
    axes[1].plot(best_rewards[cl], 'b')
    axes[1].legend(["TS", "UCB", "Clairvoyant"])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Instantaneous reward")

    axes[2].set_title(f'Cumulative regret plot {cl}')
    axes[2].plot(cumulative_regret_ts_mean[cl], 'r')
    axes[2].plot(cumulative_regret_ucb_mean[cl], 'g')
    axes[2].legend(["TS", "UCB"])
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("Cumulative regret")

    axes[3].set_title(f'Cumulative reward plot {cl}')
    axes[3].plot(cumulative_reward_ts_mean[cl], 'r')
    axes[3].plot(cumulative_reward_ucb_mean[cl], 'g')
    axes[3].plot(np.cumsum(best_rewards[cl]), 'b')
    axes[3].legend(["TS", "UCB", "Clairvoyant"])
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("Cumulative reward")
    plt.show()

# Plot the results for TS with std
for cl in classes:
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    axes[0].set_title(f'Instantaneous regret plot for TS {cl}')
    axes[0].plot(regret_ts_mean[cl], 'r')
    axes[0].fill_between(range(0, T), regret_ts_mean[cl] - regret_ts_std[cl], regret_ts_mean[cl] + regret_ts_std[cl], color='r', alpha=0.4)
    axes[0].axhline(y=0, color='b', linestyle='--')
    axes[0].legend(["TS mean", "TS std"])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Instantaneous regret")

    axes[1].set_title(f'Instantaneous reward plot for TS {cl}')
    axes[1].plot(reward_ts_mean[cl], 'r')
    axes[1].fill_between(range(0, T), reward_ts_mean[cl] - reward_ts_std[cl], reward_ts_mean[cl] + reward_ts_std[cl], color='r', alpha=0.4)
    axes[1].plot(best_rewards[cl], 'b')
    axes[1].legend(["TS mean", "TS std", "Clairvoyant"])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Instantaneous reward")

    axes[2].set_title(f'Cumulative regret plot for TS {cl}')
    axes[2].plot(cumulative_regret_ts_mean[cl], 'r')
    axes[2].fill_between(range(0, T), cumulative_regret_ts_mean[cl] - cumulative_regret_ts_std[cl], cumulative_regret_ts_mean[cl] + cumulative_regret_ts_std[cl], color='r', alpha=0.4)
    axes[2].legend(["TS mean", "TS std"])
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("Cumulative regret")

    axes[3].set_title(f'Cumulative reward plot for TS {cl}')
    axes[3].plot(cumulative_reward_ts_mean[cl], 'r')
    axes[3].fill_between(range(0, T), cumulative_reward_ts_mean[cl] - cumulative_reward_ts_std[cl], cumulative_reward_ts_mean[cl] + cumulative_reward_ts_std[cl], color='r', alpha=0.4)
    axes[3].plot(np.cumsum(best_rewards[cl]), 'b')
    axes[3].legend(["TS mean", "TS std", "Clairvoyant"])
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("Cumulative reward")
    plt.show()

# Plot the results for UCB with std
for cl in classes:
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    axes[0].set_title(f'Instantaneous regret plot for UCB {cl}')
    axes[0].plot(regret_ucb_mean[cl], 'g')
    axes[0].fill_between(range(0, T), regret_ucb_mean[cl] - regret_ucb_std[cl], regret_ucb_mean[cl] + regret_ucb_std[cl], color='g', alpha=0.4)
    axes[0].axhline(y=0, color='b', linestyle='--')
    axes[0].legend(["UCB mean", "UCB std"])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Instantaneous regret")

    axes[1].set_title(f'Instantaneous reward plot for UCB {cl}')
    axes[1].plot(reward_ucb_mean[cl], 'g')
    axes[1].fill_between(range(0, T), reward_ucb_mean[cl] - reward_ucb_std[cl], reward_ucb_mean[cl] + reward_ucb_std[cl], color='g', alpha=0.4)
    axes[1].plot(best_rewards[cl], 'b')
    axes[1].legend(["UCB mean", "UCB std", "Clairvoyant"])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Instantaneous reward")

    axes[2].set_title(f'Cumulative regret plot for UCB {cl}')
    axes[2].plot(cumulative_regret_ucb_mean[cl], 'g')
    axes[2].fill_between(range(0, T), cumulative_regret_ucb_mean[cl] - cumulative_regret_ucb_std[cl], cumulative_regret_ucb_mean[cl] + cumulative_regret_ucb_std[cl], color='g', alpha=0.4)
    axes[2].legend(["UCB mean", "UCB std"])
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("Cumulative regret")

    axes[3].set_title(f'Cumulative reward plot for UCB {cl}')
    axes[3].plot(cumulative_reward_ucb_mean[cl], 'g')
    axes[3].fill_between(range(0, T), cumulative_reward_ucb_mean[cl] - cumulative_reward_ucb_std[cl], cumulative_reward_ucb_mean[cl] + cumulative_reward_ucb_std[cl], color='g', alpha=0.4)
    axes[3].plot(np.cumsum(best_rewards[cl]), 'b')
    axes[3].legend(["UCB mean", "UCB std", "Clairvoyant"])
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("Cumulative reward")

    plt.show()

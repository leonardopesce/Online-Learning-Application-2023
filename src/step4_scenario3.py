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

# Defining the 3 categories
categories = ['C1', 'C2', 'C3']

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.2, 0.25, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([100, 2]),
                  'C2': np.array([90, 2]),
                  'C3': np.array([80, 3])}
bids_to_cum_costs = {'C1': np.array([1000, 0.07]),
                     'C2': np.array([800, 0.05]),
                     'C3': np.array([800, 0.04])}
other_costs = 400

# Bids setup
n_bids = 100
min_bid = 0.5
max_bid = 15.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma = 2

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 30

# To evaluate which are the most played prices and bids by the TS learner
ts_best_price = []
ts_best_bid = []
# To evaluate which are the most played prices and bids by the UCB learner
ucb_best_price = []
ucb_best_bid = []

# Define the environment
env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = {category: Clairvoyant(env) for category in categories}
# Optimize the problem TODO massimizzare l'aggregate model
#best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
# Optimizing the problem for all the classes separately
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = {cl: [] for cl in categories}, {cl: [] for cl in categories}, {cl: [] for cl in categories}, {cl: [] for cl in categories}, {cl: [] for cl in categories}
for category in categories:
    best_price_idx[category], best_price[category], best_bid_idx[category], best_bid[category], best_reward[category] = clairvoyant[category].maximize_reward(category)
best_reward = np.sum(np.array([best_reward[category] for category in categories]))
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
    # TS learners
    # TODO prices uguali per tutti??? qua altrimenti che priceses scelgo, per ora scelgo quelli della classe 1
    ts_learner = TSLearnerPricingAdvertising(env.prices['C1'], env.bids)
    # UCB learners
    ucb_learner = UCBLearnerPricingAdvertising(env.prices['C1'], env.bids)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        price_idx, bid_idx = ts_learner.pull_arm(env.other_costs)
        # Simulating the environment with 3 classes unknown to the learner
        bernoulli_realizations, n_clicks, cum_daily_cost = env.round_all_categories_merged(price_idx, bid_idx)
        reward = env.get_reward('C1', price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
        ts_learner.update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

        # UCB Learner
        price_idx, bid_idx = ucb_learner.pull_arm(env.other_costs)
        # Simulating the environment with 3 classes unknown to the learner
        bernoulli_realizations, n_clicks, cum_daily_cost = env.round_all_categories_merged(price_idx, bid_idx)
        reward = env.get_reward('C1', price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
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
print('Best price found in the experiments by TS')
print('The format is price: number of experiments in which it is the most played price')
print(Counter(ts_best_price))
print('Best bid found in the experiments by TS')
print('The format is bid: number of experiments in which it is the most bid price')
print(Counter(ts_best_bid))
# Print occurrences of best arm in UCB1
print('Best price found in the experiments by UCB')
print('The format is price: number of experiments in which it is the most played price')
print(Counter(ucb_best_price))
print('Best bid found in the experiments by UCB')
print('The format is bid: number of experiments in which it is the most bid price')
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

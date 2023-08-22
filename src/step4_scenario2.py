from matplotlib import pyplot as plt

from Environment import Environment
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising
from collections import Counter
import numpy as np

from MultiContextEnvironment import MultiContextEnvironment
from ContextGeneratorLearner import ContextGeneratorLearner
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

# Considered categories
categories = ['C1', 'C2', 'C3']
# Considered features and values (binary)
feature_names = ['age', 'sex']
feature_values = {'age': [0, 1], 'sex': [0, 1]}
# age: 0 -> young, 1 -> old; sex: 0 -> woman, 1 -> man
feature_values_to_categories = {(0, 0): 'C3', (0, 1): 'C1', (1, 0): 'C3', (1, 1): 'C2'}
probability_feature_values_in_categories = {'C1': {(0, 1): 1}, 'C2': {(1, 1): 1}, 'C3': {(0, 0): 0.5, (1, 0): 0.5}}


# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.1, 0.15, 0.93, 0.8, 0.5]),
                 'C2': np.array([0.05, 0.2, 0.5, 0.8, 0.93]),
                 'C3': np.array([0.93, 0.9, 0.4, 0.24, 0.01])}
bids_to_clicks = {'C1': np.array([100, 2]),
                  'C2': np.array([90, 2]),
                  'C3': np.array([80, 3])}
bids_to_cum_costs = {'C1': np.array([400, 0.035]),
                     'C2': np.array([200, 0.07]),
                     'C3': np.array([300, 0.04])}
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
n_experiments = 2
time_between_context_generation = 14


# To evaluate which are the most played prices and bids
ts_best_price, ts_best_bid, ucb_best_price, ucb_best_bid = [], [], [], []
TS, UCB = 0, 1

# Define the environment
env = MultiContextEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs,
                              categories, feature_names, feature_values, feature_values_to_categories,
                              probability_feature_values_in_categories)

env.plot_rewards(categories=['C1', 'C2', 'C3'], plot_aggregate_model=True)
env.plot_whole_advertising_model()

# Define the clairvoyant
clairvoyant = Clairvoyant(env)

best_reward = 0
for category in categories:
    _, _, _, _, a = clairvoyant.maximize_reward(category)
    best_reward += a

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# Store some features for each experiment for the learners
gpts_clicks_per_experiment = 0
gpts_mean_clicks_per_experiment = 0
gpts_sigmas_clicks_per_experiment = 0
gpts_cum_costs_per_experiment = 0
gpts_mean_cum_costs_per_experiment = 0
gpts_sigmas_cum_costs_per_experiment = 0
gpts_pulled_bids_per_experiment = 0

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    ts_context_learner = ContextGeneratorLearner(env.prices['C1'], env.bids, env.feature_name, env.feature_values,
                                                 time_between_context_generation, "TS", other_costs)
    ucb_context_learner = ContextGeneratorLearner(env.prices['C1'], env.bids, env.feature_name, env.feature_values,
                                                  time_between_context_generation, "UCB", other_costs)
    context_learners_type = [ts_context_learner, ucb_context_learner]

    # Iterate over the number of rounds
    for t in range(0, T):
        # Apply the context generation algorithm offline every 2 weeks (i.e. t multiple of 14).
        if t % time_between_context_generation == 0 and t != 0:
            print("--------------------------------------------")
            print("IT'S TIME TO UPDATE THE CONTEXT")
            print(f"TIME: {t}")
            for clt in context_learners_type:
                print("--------------------------------------------")
                clt.update_context()

        # Iterate over TS and UCB
        for clt in context_learners_type:
            # pull all the arm of the context generator
            context_price_bid_learners = clt.pull_arm(env.other_costs)
            # create variable to update the context learner
            features_list, pulled_price_list, bernoulli_realizations_list, pulled_bid_list = [], [], [], []
            clicks_given_bid_list, cost_given_bid_list, rewards = [], [], []

            # iterate over the generated contexts
            for context, price_idx, bid_idx in context_price_bid_learners:
                feature_list, bernoulli_realizations, n_clicks, cum_daily_cost = env.round(price_idx, bid_idx, context)

                # TODO: may be still based on category and not on features
                reward = []
                for i, feature in enumerate(feature_list):
                    reward.append(env.get_reward(feature, price_idx, float(np.mean(bernoulli_realizations[i])), n_clicks[i], cum_daily_cost[i]))

                # prepare data for update of context learner
                features_list.append(feature_list)
                bernoulli_realizations_list.append(bernoulli_realizations)
                pulled_price_list.append(price_idx)
                pulled_bid_list.append(bid_idx)
                clicks_given_bid_list.append(n_clicks)
                cost_given_bid_list.append(cum_daily_cost)
                rewards.append(reward)

            clt.update(pulled_price_list=pulled_price_list, bernoulli_realizations_list=bernoulli_realizations_list,
                       features_list=features_list, pulled_bid_list=pulled_bid_list,
                       clicks_given_bid_list=clicks_given_bid_list, cost_given_bid_list=cost_given_bid_list,
                       rewards=rewards)

    # Store the most played prices and bids by TS
    #ts_best_price.append(Counter(context_learners_type[TS].get_pulled_prices()).most_common(1)[0])
    #ts_best_bid.append(Counter(context_learners_type[TS].get_pulled_bids()).most_common(1)[0])

    # Store the most played prices and bids by UCB1
    #ucb_best_price.append(Counter(context_learners_type[UCB].get_pulled_prices()).most_common(1)[0])
    #ucb_best_bid.append(Counter(context_learners_type[UCB].get_pulled_bids()).most_common(1)[0])

    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(np.array(context_learners_type[TS].get_collective_reward()))
    ucb_reward_per_experiment.append(np.array(context_learners_type[UCB].get_collective_reward()))

    """gpts_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.collected_clicks)
    gpts_mean_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.means_clicks)
    gpts_sigmas_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.sigmas_clicks)
    gpts_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.collected_costs)
    gpts_mean_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.means_costs)
    gpts_sigmas_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.sigmas_costs)
    gpts_pulled_bids_per_experiment[category].append(ts_learner[category].GPTS_advertising.pulled_bids)"""

# Print occurrences of best arm in TS
#print(Counter(ts_best_price))
#print(Counter(ts_best_bid))
# Print occurrences of best arm in UCB1
#print(Counter(ucb_best_price))
#print(Counter(ucb_best_bid))

#plot data
best_rewards = np.ones((T,)) * best_reward
regret_ts_mean = np.mean(best_rewards - ts_reward_per_experiment, axis=0)
regret_ts_std = np.std(best_rewards - ts_reward_per_experiment, axis=0)
regret_ucb_mean = np.mean(best_reward - ucb_reward_per_experiment, axis=0)
regret_ucb_std = np.std(best_rewards - ucb_reward_per_experiment, axis=0)
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

# Plot the results, comparison TS-UCB
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
axes[0].set_title(f'Instantaneous regret plot')
axes[0].plot(regret_ts_mean, 'r')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS", "UCB"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title(f'Instantaneous reward plot')
axes[1].plot(reward_ts_mean, 'r')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS", "UCB", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title(f'Cumulative regret plot')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].legend(["TS", "UCB"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title(f'Cumulative reward plot')
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

axes[0].set_title(f'Instantaneous regret plot for TS')
axes[0].plot(regret_ts_mean, 'r')
axes[0].fill_between(range(0, T), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS mean", "TS std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title(f'Instantaneous reward plot for TS')
axes[1].plot(reward_ts_mean, 'r')
axes[1].fill_between(range(0, T), reward_ts_mean - reward_ts_std, reward_ts_mean + reward_ts_std, color='r', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS mean", "TS std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title(f'Cumulative regret plot for TS')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].fill_between(range(0, T), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='r', alpha=0.4)
axes[2].legend(["TS mean", "TS std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title(f'Cumulative reward plot for TS')
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

axes[0].set_title(f'Instantaneous regret plot for UCB')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].fill_between(range(0, T), regret_ucb_mean - regret_ucb_std, regret_ucb_mean + regret_ucb_std, color='g', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["UCB mean", "UCB std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title(f'Instantaneous reward plot for UCB')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].fill_between(range(0, T), reward_ucb_mean - reward_ucb_std, reward_ucb_mean + reward_ucb_std, color='g', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title(f'Cumulative regret plot for UCB')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].fill_between(range(0, T), cumulative_regret_ucb_mean - cumulative_regret_ucb_std, cumulative_regret_ucb_mean + cumulative_regret_ucb_std, color='g', alpha=0.4)
axes[2].legend(["UCB mean", "UCB std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title(f'Cumulative reward plot for UCB')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].fill_between(range(0, T), cumulative_reward_ucb_mean - cumulative_reward_ucb_std, cumulative_reward_ucb_mean + cumulative_reward_ucb_std, color='g', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")

plt.show()

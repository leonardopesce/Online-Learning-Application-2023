from collections import Counter

import numpy as np

from tqdm import tqdm

import settings

from Environments import Environment
from Learners import Clairvoyant, TSLearnerPricingAdvertising, UCBLearnerPricingAdvertising
from Utilities import plot_all_algorithms, plot_clicks_curve, plot_costs_curve, plot_all_algorithms_divided

"""
Step 4: Contexts and their generation
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

bids = np.linspace(settings.min_bid, settings.max_bid, settings.n_bids)
# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 100

algorithms = ['UCB', 'TS']

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
environments = dict()
rewards = {algorithm: {category: [] for category in categories} for algorithm in algorithms}
best_rewards = dict()

# To store the learners to plot the advertising curves
gp_learners = {algorithm: {category: [] for category in categories} for algorithm in algorithms}

# To evaluate which are the most played prices and bids
best_prices = {algorithm: {category: [] for category in categories} for algorithm in algorithms}
best_bids = {algorithm: {category: [] for category in categories} for algorithm in algorithms}

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = {category: Clairvoyant(env) for category in categories}
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = {category: [] for category in categories}, {category: [] for category in categories}, {category: [] for category in categories}, {category: [] for category in categories}, {category: [] for category in categories}
for category in categories:
    best_price_idx[category], best_price[category], best_bid_idx[category], best_bid[category], best_reward[category] = clairvoyant[category].maximize_reward(category)
    best_rewards[category] = np.ones((T,)) * best_reward[category]

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environments
    environments = {algorithm: Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs) for algorithm in algorithms}

    # Define the learners
    learners['UCB'] = {category: UCBLearnerPricingAdvertising(settings.prices[category], env.bids, sklearn=False) for
                       category in categories}
    learners['TS'] = {category: TSLearnerPricingAdvertising(settings.prices[category], env.bids, sklearn=False) for
                      category in categories}

    # Iterate over the number of rounds
    for t in range(0, T):
        for algorithm in algorithms:
            for category in categories:
                price_idx, bid_idx = learners[algorithm][category].pull_arm(environments[algorithm].other_costs)
                bernoulli_realizations, n_clicks, cum_daily_cost = environments[algorithm].round(category, price_idx, bid_idx)
                reward = environments[algorithm].get_reward(category, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
                learners[algorithm][category].update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

    # Store the most played prices and bids, the values of the collected rewards of the learners and the learners
    for algorithm in algorithms:
        for category in categories:
            best_prices[algorithm][category].append(Counter(learners[algorithm][category].get_pulled_prices()).most_common(1)[0][0])
            best_bids[algorithm][category].append(Counter(learners[algorithm][category].get_pulled_bids()).most_common(1)[0][0])
            rewards[algorithm][category].append(learners[algorithm][category].learner_pricing.collected_rewards)
            gp_learners[algorithm][category].append(learners[algorithm][category].GP_advertising)

# Print occurrences of best arm
for algorithm in algorithms:
    for category in categories:
        print('Best price found in the experiments by ' + algorithm + ' for category ' + category)
        print('The format is price: number of experiments in which it is the most played price')
        settings.iterate_over_counter(Counter(best_prices[algorithm][category]), env.prices[category])
        print('Best bid found in the experiments by ' + algorithm + ' for category ' + category)
        print('The format is bid: number of experiments in which it is the most bid price')
        settings.iterate_over_counter(Counter(best_bids[algorithm][category]), env.bids)

# Plot the results
for category in categories:
    reward_per_algorithm = [rewards[algorithm][category] for algorithm in algorithms]
    labels = [f'{algorithm} - {category}' for algorithm in algorithms]
    plot_all_algorithms(reward_per_algorithm, best_rewards[category], np.arange(0, T, 1), labels)

#for i, algorithm in enumerate(algorithms):
#    for category in categories:
#        plot_single_algorithm(rewards[algorithm][category], best_rewards[category], f'{algorithm} - {category}', np.arange(0, T, 1))

for category in categories:
    gp_learners_category = {algorithm: gp_learners[algorithm][category] for algorithm in algorithms}
    plot_clicks_curve(bids, gp_learners_category, algorithms, original=env.get_clicks_curve(bids, category), additional_label=f' - {category}')
    plot_costs_curve(bids, gp_learners_category, algorithms, original=env.get_costs_curve(bids, category), additional_label=f' - {category}')

# Plot the aggregated results
total_best_rewards = np.sum(np.array([best_rewards[category] for category in categories]), axis=0)
total_reward_per_algorithm = [np.sum(np.array([rewards[algorithm][category] for category in categories]), axis=0) for algorithm in algorithms]
plot_all_algorithms(total_reward_per_algorithm, total_best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_1")
plot_all_algorithms_divided(total_reward_per_algorithm, total_best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_1")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(total_reward_per_algorithm[i], total_best_rewards, f'Aggregated {algorithm}', np.arange(0, T, 1))
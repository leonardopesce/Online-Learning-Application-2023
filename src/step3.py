from Environment import Environment
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising
from collections import Counter
import numpy as np
from plots import plot_single_algorithm, plot_all_algorithms, plot_clicks_curve, plot_costs_curve
import settings

"""
Simulation for the step 3: Learning for joint pricing and advertising

Consider the case in which all the users belong to class C1, and no information about the advertising and pricing curves
is known beforehand. Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves, reporting
the plots of the average (over a sufficiently large number of runs) value and standard deviation of the cumulative 
regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

bids = np.linspace(settings.min_bid, settings.max_bid, settings.n_bids)
# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 10

algorithms = ['UCB', 'TS']
# To store the learners to plot the advertising curves
gp_learners = {algorithm: [] for algorithm in algorithms}

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# To evaluate which are the most played prices and bids
ts_best_price = []
ts_best_bid = []
ucb_best_price = []
ucb_best_bid = []

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
best_rewards = np.ones((T,)) * best_reward

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
    ts_reward_per_experiment.append(ts_learner.learner_pricing.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.learner_pricing.collected_rewards)

    # Store the learners
    gp_learners['TS'].append(ts_learner.GP_advertising)
    gp_learners['UCB'].append(ucb_learner.GP_advertising)

def iterate_over_counter(counter, reference_array):
    for key, value in counter.items():
        print(f"{reference_array[key]}, index {key}, is the best in {value} experiments")

# Print occurrences of best arm in TS
print('Best price found in the experiments by TS')
print('The format is price: number of experiments in which it is the most played price')
iterate_over_counter(Counter(ts_best_price), env.prices[category])
print('Best bid found in the experiments by TS')
print('The format is bid: number of experiments in which it is the most bid price')
iterate_over_counter(Counter(ts_best_bid), env.bids)
# Print occurrences of best arm in UCB1
print('Best price found in the experiments by UCB')
print('The format is price: number of experiments in which it is the most played price')
iterate_over_counter(Counter(ucb_best_price), env.prices[category])
print('Best bid found in the experiments by UCB')
print('The format is bid: number of experiments in which it is the most bid price')
iterate_over_counter(Counter(ucb_best_bid), env.bids)

# Plot the results
reward_per_algorithm = [ucb_reward_per_experiment, ts_reward_per_experiment]
plot_clicks_curve(bids, gp_learners, algorithms)
plot_costs_curve(bids, gp_learners, algorithms)
plot_all_algorithms(reward_per_algorithm, best_rewards, algorithms)
for i, label in enumerate(algorithms):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, label, np.arange(0, T, 1))

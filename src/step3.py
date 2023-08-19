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

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
environments = dict()
rewards = {algorithm: [] for algorithm in algorithms}
best_rewards = np.array([])

# To store the learners to plot the advertising curves
gp_learners = {algorithm: [] for algorithm in algorithms}

# To evaluate which are the most played prices and bids
best_prices = {algorithm: [] for algorithm in algorithms}
best_bids = {algorithm: [] for algorithm in algorithms}

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
best_rewards = np.append(best_rewards, np.ones((T,)) * best_reward)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environments
    environments = {algorithm: Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs) for algorithm in algorithms}

    # Define the learners
    learners['UCB'] = UCBLearnerPricingAdvertising(settings.prices[category], env.bids)
    learners['TS'] = TSLearnerPricingAdvertising(settings.prices[category], env.bids)

    # Iterate over the number of rounds
    for t in range(0, T):
        for algorithm in algorithms:
            price_idx, bid_idx = learners[algorithm].pull_arm(environments[algorithm].other_costs)
            bernoulli_realizations, n_clicks, cum_daily_cost = environments[algorithm].round(category, price_idx, bid_idx)
            reward = env.get_reward(category, price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
            learners[algorithm].update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

    # Store the most played prices and bids, the values of the collected rewards of the learners and the learners
    for algorithm in algorithms:
        best_prices[algorithm].append(Counter(learners[algorithm].get_pulled_prices()).most_common(1)[0][0])
        best_bids[algorithm].append(Counter(learners[algorithm].get_pulled_bids()).most_common(1)[0][0])
        rewards[algorithm].append(learners[algorithm].learner_pricing.collected_rewards)
        gp_learners[algorithm].append(learners[algorithm].GP_advertising)


def iterate_over_counter(counter, reference_array):
    for key, value in counter.items():
        print(f"{reference_array[key]}, index {key}, is the best in {value} experiments")


# Print occurrences of best arm in TS
for algorithm in algorithms:
    print('Best price found in the experiments by ' + algorithm)
    print('The format is price: number of experiments in which it is the most played price')
    iterate_over_counter(Counter(best_prices[algorithm]), env.prices[category])
    print('Best bid found in the experiments by ' + algorithm)
    print('The format is bid: number of experiments in which it is the most bid price')
    iterate_over_counter(Counter(best_bids[algorithm]), env.bids)

# Plot the results
reward_per_algorithm = [rewards[algorithm] for algorithm in algorithms]
plot_clicks_curve(bids, gp_learners, algorithms, original=env.get_clicks_curve(bids, category))
plot_costs_curve(bids, gp_learners, algorithms, original=env.get_costs_curve(bids, category))
plot_all_algorithms(reward_per_algorithm, best_rewards, algorithms)
for i, algorithm in enumerate(algorithms):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

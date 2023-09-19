from Environment import Environment
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from TSPricingAdvertising import TSLearnerPricingAdvertising
from UCBPricingAdvertising import UCBLearnerPricingAdvertising
from collections import Counter
import numpy as np
from plots import plot_single_algorithm, plot_all_algorithms, plot_all_algorithms_divided
import settings

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

# Defining the 3 categories
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
rewards = {algorithm: [] for algorithm in algorithms}
best_rewards = np.array([])

# To evaluate which are the most played prices and bids
best_prices = {algorithm: [] for algorithm in algorithms}
best_bids = {algorithm: [] for algorithm in algorithms}

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

# Optimizing the problem for all the classes separately
best_reward = 0
for category in categories:
    _, _, _, _, best_reward_category = clairvoyant.maximize_reward(category)
    best_reward += best_reward_category
best_rewards = np.append(best_rewards, np.ones((T,)) * best_reward)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environments
    environments = {algorithm: Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs) for algorithm in algorithms}

    # Define the learners, the prices are the same for all categories, so I can pass C1
    learners['UCB'] = UCBLearnerPricingAdvertising(settings.prices['C1'], env.bids, sklearn=False)
    learners['TS'] = TSLearnerPricingAdvertising(settings.prices['C1'], env.bids, sklearn=False)

    # Iterate over the number of rounds
    for t in range(0, T):
        for algorithm in algorithms:
            price_idx, bid_idx = learners[algorithm].pull_arm(environments[algorithm].other_costs)
            # Simulating the environment with 3 classes unknown to the learner
            bernoulli_realizations, n_clicks, cum_daily_cost = environments[algorithm].round_all_categories_merged(price_idx, bid_idx)
            reward = environments[algorithm].get_reward('C1', price_idx, np.mean(bernoulli_realizations), n_clicks, cum_daily_cost)
            learners[algorithm].update(price_idx, bernoulli_realizations, bid_idx, n_clicks, cum_daily_cost, reward)

    # Store the most played prices and bids, the values of the collected rewards of the learners and the learners
    for algorithm in algorithms:
        best_prices[algorithm].append(Counter(learners[algorithm].get_pulled_prices()).most_common(1)[0][0])
        best_bids[algorithm].append(Counter(learners[algorithm].get_pulled_bids()).most_common(1)[0][0])
        rewards[algorithm].append(learners[algorithm].learner_pricing.collected_rewards)

# Print occurrences of best arm
for algorithm in algorithms:
    print('Best price found in the experiments by ' + algorithm)
    print('The format is price: number of experiments in which it is the most played price')
    print(Counter(best_prices[algorithm]))
    print('Best bid found in the experiments by ' + algorithm)
    print('The format is bid: number of experiments in which it is the most bid price')
    print(Counter(best_bids[algorithm]))

# Plot the results
reward_per_algorithm = [rewards[algorithm] for algorithm in algorithms]
plot_all_algorithms(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_3")
plot_all_algorithms_divided(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_3")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

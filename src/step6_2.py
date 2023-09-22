import numpy as np

from tqdm import tqdm

import settings

from Learners import Clairvoyant, UCBLearner, SWUCBLearner, CUSUMUCBLearner, EXP3Learner
from Environments import NonStationaryEnvironment
from Utilities import plot_all_algorithms, plot_all_algorithms_divided

"""
Simulation for the step 6: dealing with non-stationary environments with many abrupt changes

Develop the EXP3 algorithm, which is devoted to dealing with adversarial settings. This algorithm can be also used to 
deal with non-stationary settings when no information about the specific form of non-stationarity is known beforehand. 
Consider a simplified version of Step 5 in which the bid is fixed. First, apply the EXP3 algorithm to this setting. 
The expected result is that EXP3 performs worse than the two non-stationary versions of UCB1. Subsequently, consider a 
different non-stationary setting with a higher non-stationarity degree. Such a degree can be modeled by having a large 
number of phases that frequently change. In particular, consider 5 phases, each one associated with a different optimal 
price, and these phases cyclically change with a high frequency. In this new setting, apply EXP3, UCB1, and the two 
non-stationary flavors of UBC1. The expected result is that EXP3 outperforms the non-stationary version of UCB1 in this 
setting.
"""

# Considered category is C1
category = 'C1'

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 100

algorithms = ['UCB', 'SW-UCB', 'CUSUM-UCB', 'EXP3']

# Learners parameters
print(f"The window size is {settings.window_size}")
print(f"Parameters for CUSUM-UCB are M={settings.M}, eps={settings.eps}, h={round(settings.h, 2)}, alpha={round(settings.alpha, 2)}")

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
environments = dict()
rewards = {algorithm: [] for algorithm in algorithms}
best_rewards = np.array([])

# Define the environment
env = NonStationaryEnvironment(settings.n_prices, settings.prices_step6, settings.probabilities_step6, settings.bids_to_clicks_cost_step6, settings.bids_to_cum_costs_cost_step6, settings.other_costs, settings.phases_duration_step6)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

best_reward_per_phase = []
# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(settings.phases_duration_step6):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_reward = clairvoyant.maximize_reward_given_bid('C' + str(phase + 1), settings.bid_idx)
    best_reward_per_phase.append(best_reward)

# Save the best rewards along the year
for t in range(T):
    phase_idx = np.searchsorted(np.cumsum(settings.phases_duration_step6), t % np.sum(settings.phases_duration_step6), side='right')
    best_rewards = np.append(best_rewards, best_reward_per_phase[phase_idx])

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environment and learners

    # Define the environments
    environments = {algorithm: NonStationaryEnvironment(settings.n_prices, settings.prices_step6, settings.probabilities_step6, settings.bids_to_clicks_cost_step6, settings.bids_to_cum_costs_cost_step6, settings.other_costs, settings.phases_duration_step6) for algorithm in algorithms}

    # Define the learners
    learners['UCB'] = UCBLearner(settings.prices_step6[category])
    learners['SW-UCB'] = SWUCBLearner(settings.prices_step6[category], settings.window_size)
    learners['CUSUM-UCB'] = CUSUMUCBLearner(settings.prices_step6[category], M=settings.M, eps=settings.eps, h=settings.h, alpha=settings.alpha)
    learners['EXP3'] = EXP3Learner(settings.prices_step6[category], worst_reward=0, best_reward=max(settings.prices_step6[category]) - settings.other_costs, gamma=0.2, other_costs=settings.other_costs)

    # Iterate over the number of rounds
    for t in range(0, T):
        for algorithm in algorithms:
            n_clicks = environments[algorithm].get_n_clicks(category, settings.bid_idx)
            cum_daily_costs = environments[algorithm].get_cum_daily_costs(category, settings.bid_idx)
            pulled_arm = learners[algorithm].pull_arm(settings.other_costs, n_clicks, cum_daily_costs)
            bernoulli_realizations = environments[algorithm].round_pricing(pulled_arm, int(np.floor(n_clicks)))
            reward = environments[algorithm].get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), settings.bid_idx)
            learners[algorithm].update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    for algorithm in algorithms:
        rewards[algorithm].append(learners[algorithm].collected_rewards)

# Plot the results
reward_per_algorithm = [rewards[algorithm] for algorithm in algorithms]
plot_all_algorithms(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step6_2")
plot_all_algorithms_divided(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step6_2")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

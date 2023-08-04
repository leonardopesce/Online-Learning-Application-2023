from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
from EXP3 import EXP3Learner
from plots import plot_single_algorithm, plot_all_algorithms

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

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700]),
          'C4': np.array([500, 550, 600, 650, 700]),
          'C5': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.03, 0.04, 0.05, 0.03, 0.01]),  # best arm is 2 (starting from 0)
                 'C2': np.array([0.03, 0.05, 0.03, 0.05, 0.02]),  # best arm is 3
                 'C3': np.array([0.06, 0.07, 0.02, 0.02, 0.01]),  # best arm is 1
                 'C4': np.array([0.03, 0.04, 0.04, 0.02, 0.05]),  # best arm is 4
                 'C5': np.array([0.1, 0.03, 0.03, 0.02, 0.01])}   # best arm is 0
bids_to_clicks = {'C1': np.array([100, 2]),  # this curve doesn't change
                  'C2': np.array([100, 2]),
                  'C3': np.array([100, 2]),
                  'C4': np.array([100, 2]),
                  'C5': np.array([100, 2])}
bids_to_cum_costs = {'C1': np.array([20, 0.5]),  # this curve doesn't change
                     'C2': np.array([20, 0.5]),
                     'C3': np.array([20, 0.5]),
                     'C4': np.array([20, 0.5]),
                     'C5': np.array([20, 0.5])}

other_costs = 400
phases_duration = [11, 21, 23, 15, 19]
#phases_duration = [5, 11, 8, 9, 6]
bid_idx = 20  #TODO, bid is fixed, is okay like this?
# with this bid_idx the n_clicks=100 and cum_daily_costs=16.4

# Time horizon of the experiment
T = 365
window_size = 50
# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 200
# Store the rewards for each experiment for the learners
ucb_reward_per_experiment = []
swucb_reward_per_experiment = []
cusum_ucb_reward_per_experiment = []
exp3_reward_per_experiment = []
best_rewards = np.array([])

# Define the environment
env = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

best_reward_per_phase = []
# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(phases_duration):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_reward = clairvoyant.maximize_reward_given_bid('C' + str(phase + 1), bid_idx)
    best_reward_per_phase.append(best_reward)

# Save the best rewards along the year
for t in range(T):
    phase_idx = np.searchsorted(np.cumsum(phases_duration), t % np.sum(phases_duration), side='right')
    best_rewards = np.append(best_rewards, best_reward_per_phase[phase_idx])

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environment and learners

    # UCB1
    env_ucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    ucb_learner = UCBLearner(prices[category])

    # SW-UCB
    env_swucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    swucb_learner = SWUCBLearner(prices[category], window_size)

    # CUSUM-UCB
    env_cusum_ucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    cusum_ucb_learner = CUSUMUCBLearner(prices[category])

    # EXP3
    env_exp3 = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    n_clicks = env_exp3.get_n_clicks(category, bid_idx)
    cum_daily_costs = env_exp3.get_cum_daily_costs(category, bid_idx)
    #worst_reward = n_clicks * 0 * (min(prices[category]) - other_costs) - cum_daily_costs
    #best_reward = n_clicks * 1 * (max(prices[category]) - other_costs) - cum_daily_costs
    worst_reward = 0
    best_reward = 300
    exp3_learner = EXP3Learner(prices[category], worst_reward, best_reward, gamma=0.1, other_costs=other_costs)

    # Iterate over the number of rounds
    for t in range(0, T):
        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        n_clicks = env_ucb.get_n_clicks(category, bid_idx)
        bernoulli_realizations = env_ucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), bid_idx)
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # SW-UCB Learner
        pulled_arm = swucb_learner.pull_arm()
        n_clicks = env_swucb.get_n_clicks(category, bid_idx)
        bernoulli_realizations = env_swucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_swucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), bid_idx)
        swucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # CUSUM-UCB Learner
        pulled_arm = cusum_ucb_learner.pull_arm()
        n_clicks = env_cusum_ucb.get_n_clicks(category, bid_idx)
        bernoulli_realizations = env_cusum_ucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_cusum_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), bid_idx)
        cusum_ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # EXP3 Learner
        pulled_arm = exp3_learner.pull_arm()
        n_clicks = env_exp3.get_n_clicks(category, bid_idx)
        bernoulli_realizations = env_exp3.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_exp3.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), bid_idx)
        exp3_learner.update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    swucb_reward_per_experiment.append(swucb_learner.collected_rewards)
    cusum_ucb_reward_per_experiment.append(cusum_ucb_learner.collected_rewards)
    exp3_reward_per_experiment.append(exp3_learner.collected_rewards)

# Plot the results
reward_per_algorithm = [ucb_reward_per_experiment, swucb_reward_per_experiment, cusum_ucb_reward_per_experiment, exp3_reward_per_experiment]
labels = ['UCB', 'SW-UCB', 'CUSUM-UCB', 'EXP3']

plot_all_algorithms(reward_per_algorithm, best_rewards, labels)
for i, label in enumerate(labels):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, label, np.arange(0, T, 1))

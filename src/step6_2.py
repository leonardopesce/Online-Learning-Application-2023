from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
from EXP3 import EXP3Learner
from plots import plot_single_algorithm, plot_all_algorithms
import settings

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
window_size = int(3 * (T ** 0.5))
M = 150
eps = 0.1
h = 2 * np.log(T)
alpha = 0.1
# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 50
# Store the rewards for each experiment for the learners
ucb_reward_per_experiment = []
swucb_reward_per_experiment = []
cusum_ucb_reward_per_experiment = []
exp3_reward_per_experiment = []
best_rewards = np.array([])

# Define the environment
env = NonStationaryEnvironment(settings.n_prices, settings.prices5, settings.probabilities5, settings.bids_to_clicks_cost5, settings.bids_to_cum_costs_cost5, settings.other_costs, settings.phases_duration5)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

best_reward_per_phase = []
# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(settings.phases_duration5):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_reward = clairvoyant.maximize_reward_given_bid('C' + str(phase + 1), settings.bid_idx)
    best_reward_per_phase.append(best_reward)

# Save the best rewards along the year
for t in range(T):
    phase_idx = np.searchsorted(np.cumsum(settings.phases_duration5), t % np.sum(settings.phases_duration5), side='right')
    best_rewards = np.append(best_rewards, best_reward_per_phase[phase_idx])

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environment and learners

    # UCB1
    env_ucb = NonStationaryEnvironment(settings.n_prices, settings.prices5, settings.probabilities5, settings.bids_to_clicks_cost5, settings.bids_to_cum_costs_cost5, settings.other_costs, settings.phases_duration5)
    ucb_learner = UCBLearner(settings.prices5[category])

    # SW-UCB
    env_swucb = NonStationaryEnvironment(settings.n_prices, settings.prices5, settings.probabilities5, settings.bids_to_clicks_cost5, settings.bids_to_cum_costs_cost5, settings.other_costs, settings.phases_duration5)
    swucb_learner = SWUCBLearner(settings.prices5[category], window_size)

    # CUSUM-UCB
    env_cusum_ucb = NonStationaryEnvironment(settings.n_prices, settings.prices5, settings.probabilities5, settings.bids_to_clicks_cost5, settings.bids_to_cum_costs_cost5, settings.other_costs, settings.phases_duration5)
    cusum_ucb_learner = CUSUMUCBLearner(settings.prices5[category], M=M, eps=eps, h=h, alpha=alpha)

    # EXP3
    env_exp3 = NonStationaryEnvironment(settings.n_prices, settings.prices5, settings.probabilities5, settings.bids_to_clicks_cost5, settings.bids_to_cum_costs_cost5, settings.other_costs, settings.phases_duration5)
    n_clicks = env_exp3.get_n_clicks(category, settings.bid_idx)
    cum_daily_costs = env_exp3.get_cum_daily_costs(category, settings.bid_idx)
    #worst_reward = n_clicks * 0 * (min(prices5[category]) - other_costs) - cum_daily_costs
    #best_reward = n_clicks * 1 * (max(prices5[category]) - other_costs) - cum_daily_costs
    worst_reward = 0
    best_reward = max(settings.prices5[category]) - settings.other_costs
    exp3_learner = EXP3Learner(settings.prices5[category], worst_reward, best_reward, gamma=0.1, other_costs=settings.other_costs)

    # Iterate over the number of rounds
    for t in range(0, T):
        # UCB Learner
        n_clicks = env_ucb.get_n_clicks(category, settings.bid_idx)
        cum_daily_costs = env_ucb.get_cum_daily_costs(category, settings.bid_idx)
        pulled_arm = ucb_learner.pull_arm(settings.other_costs, n_clicks, cum_daily_costs)
        bernoulli_realizations = env_ucb.round_pricing(pulled_arm, int(np.floor(n_clicks)))
        reward = env_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), settings.bid_idx)
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # SW-UCB Learner
        n_clicks = env_swucb.get_n_clicks(category, settings.bid_idx)
        cum_daily_costs = env_swucb.get_cum_daily_costs(category, settings.bid_idx)
        pulled_arm = swucb_learner.pull_arm(settings.other_costs, n_clicks, cum_daily_costs)
        bernoulli_realizations = env_swucb.round_pricing(pulled_arm, int(np.floor(n_clicks)))
        reward = env_swucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), settings.bid_idx)
        swucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # CUSUM-UCB Learner
        n_clicks = env_cusum_ucb.get_n_clicks(category, settings.bid_idx)
        cum_daily_costs = env_cusum_ucb.get_cum_daily_costs(category, settings.bid_idx)
        pulled_arm = cusum_ucb_learner.pull_arm(settings.other_costs, n_clicks, cum_daily_costs)
        bernoulli_realizations = env_cusum_ucb.round_pricing(pulled_arm, int(np.floor(n_clicks)))
        reward = env_cusum_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), settings.bid_idx)
        cusum_ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # EXP3 Learner
        n_clicks = env_exp3.get_n_clicks(category, settings.bid_idx)
        cum_daily_costs = env_exp3.get_cum_daily_costs(category, settings.bid_idx)
        pulled_arm = exp3_learner.pull_arm(settings.other_costs, n_clicks, cum_daily_costs)
        bernoulli_realizations = env_exp3.round_pricing(pulled_arm, int(np.floor(n_clicks)))
        reward = env_exp3.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), settings.bid_idx)
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

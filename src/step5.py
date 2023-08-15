from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
from plots import plot_single_algorithm, plot_all_algorithms
import settings

"""
Simulation for the step 5: dealing with non-stationary environments with two abrupt changes 

Consider the case in which there is a single-user class C1. Assume that the curve related to the pricing problem is 
unknown while the curves related to the advertising problems are known. 
Furthermore, consider the situation in which the curves related to pricing are non-stationary, being subject to 
seasonal phases (3 different phases spread over the time horizon). Provide motivation for the phases. 
Apply the UCB1 algorithm and two non-stationary flavors of the UCB1 algorithm defined as follows. 
The first one is passive and exploits a sliding window, while the second one is active and exploits a change detection 
test. Provide a sensitivity analysis of the parameters employed in the algorithms, evaluating different values of 
the length of the sliding window in the first case and different values for the parameters of the change detection test
 in the second case. Report the plots with the average (over a sufficiently large number of runs) value and standard 
 deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward. 
 Compare the results of the three algorithms used.
"""

# Considered category is C1
category = 'C1'

# Time horizon of the experiment
T = 365
assert np.sum(settings.phases_duration) == T

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 50

# Learners parameters
print(f"The window size is {settings.window_size}")
print(f"Parameters for CUSUM-UCB are M={settings.M}, eps={settings.eps}, h={round(settings.h, 2)}, alpha={round(settings.alpha, 2)}")

# Store the rewards for each experiment for the learners
ucb_reward_per_experiment = []
swucb_reward_per_experiment = []
cusum_ucb_reward_per_experiment = []
best_rewards = np.array([])

# Define the environment
env = NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(settings.phases_duration):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward('C' + str(phase + 1))
    best_rewards = np.append(best_rewards, [best_reward] * phase_len)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environment and learners

    # UCB1
    env_ucb = NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration)
    ucb_learner = UCBLearner(settings.prices[category])

    # SW-UCB
    env_swucb = NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration)
    swucb_learner = SWUCBLearner(settings.prices[category], settings.window_size)

    # CUSUM-UCB
    env_cusum_ucb = NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration)
    cusum_ucb_learner = CUSUMUCBLearner(settings.prices[category], M=settings.M, eps=settings.eps, h=settings.h, alpha=settings.alpha)

    # Iterate over the number of rounds
    for t in range(0, T):
        # UCB Learner
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, ucb_learner.get_conv_prob(arm) * (
                env_ucb.prices[category][arm] - env_ucb.other_costs))[0] for arm in range(settings.n_prices)]
        n_clicks_list = np.array([env_ucb.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env_ucb.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = ucb_learner.pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)

        bernoulli_realizations = env_ucb.round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
        reward = env_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # SW-UCB Learner
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, swucb_learner.get_conv_prob(arm) * (
                env_swucb.prices[category][arm] - env_swucb.other_costs))[0] for arm in range(settings.n_prices)]
        n_clicks_list = np.array([env_swucb.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env_swucb.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = swucb_learner.pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)

        bernoulli_realizations = env_swucb.round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
        reward = env_swucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
        swucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # CUSUM-UCB Learner
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, cusum_ucb_learner.get_conv_prob(arm) * (
                env_cusum_ucb.prices[category][arm] - env_cusum_ucb.other_costs))[0] for arm in range(settings.n_prices)]
        n_clicks_list = np.array([env_cusum_ucb.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env_cusum_ucb.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = cusum_ucb_learner.pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)

        bernoulli_realizations = env_cusum_ucb.round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
        reward = env_cusum_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations),
                                                 best_bids_idx[pulled_arm])
        cusum_ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    swucb_reward_per_experiment.append(swucb_learner.collected_rewards)
    cusum_ucb_reward_per_experiment.append(cusum_ucb_learner.collected_rewards)

# Plot the results
reward_per_algorithm = [ucb_reward_per_experiment, swucb_reward_per_experiment, cusum_ucb_reward_per_experiment]
labels = ['UCB', 'SW-UCB', 'CUSUM-UCB']

plot_all_algorithms(reward_per_algorithm, best_rewards, labels)
for i, label in enumerate(labels):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, label, np.arange(0, T, 1))

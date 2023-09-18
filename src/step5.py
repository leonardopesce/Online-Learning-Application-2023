from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
from plots import plot_single_algorithm, plot_all_algorithms, plot_all_algorithms_divided
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

algorithms = ['UCB', 'SW-UCB', 'CUSUM-UCB']

# Learners parameters
print(f"The window size is {settings.window_size}")
print(f"Parameters for CUSUM-UCB are M={settings.M}, eps={settings.eps}, h={round(settings.h, 2)}, alpha={round(settings.alpha, 2)}")

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
environments = dict()
rewards = {algorithm: [] for algorithm in algorithms}
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
    # Define the environments
    environments = {algorithm: NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration) for algorithm in algorithms}

    # Define the learners
    learners['UCB'] = UCBLearner(settings.prices[category])
    learners['SW-UCB'] = SWUCBLearner(settings.prices[category], settings.window_size)
    learners['CUSUM-UCB'] = CUSUMUCBLearner(settings.prices[category], M=settings.M, eps=settings.eps, h=settings.h, alpha=settings.alpha)

    # Iterate over the number of rounds
    for t in range(0, T):
        for algorithm in algorithms:
            best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, learners[algorithm].get_conv_prob(arm) * (environments[algorithm].prices[category][arm] - environments[algorithm].other_costs))[0] for arm in range(settings.n_prices)]
            n_clicks_list = np.array([environments[algorithm].get_n_clicks(category, bid) for bid in best_bids_idx])
            cum_daily_costs_list = np.array([environments[algorithm].get_cum_daily_costs(category, bid) for bid in best_bids_idx])
            pulled_arm = learners[algorithm].pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)
            bernoulli_realizations = environments[algorithm].round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
            reward = environments[algorithm].get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
            learners[algorithm].update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    for algorithm in algorithms:
        rewards[algorithm].append(learners[algorithm].collected_rewards)

# Plot the results
reward_per_algorithm = [rewards[algorithm] for algorithm in algorithms]
plot_all_algorithms(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step5")
plot_all_algorithms_divided(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step5")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

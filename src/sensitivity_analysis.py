import numpy as np
from tqdm import tqdm
from Clairvoyant import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
import settings
from plots import plot_single_algorithm, plot_all_algorithms

"""
Sensitivity analysis of the parameters employed in the algorithms, evaluating different values of the length of the 
sliding window for SW-UCB and different values for the parameters of the change detection test for CUSUM-UCB.
"""

# Set to True the algorithm you want to analyze
sw_ucb = False
cusum_ucb = True

# Considered category is C1
category = 'C1'

# Time horizon of the experiment
T = 365
assert np.sum(settings.phases_duration) == T

n_experiments = 20

# Define the environment
env = NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
best_rewards = np.array([])

# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(settings.phases_duration):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward('C' + str(phase + 1))
    best_rewards = np.append(best_rewards, [best_reward] * phase_len)

# SW-UCB analysis
if sw_ucb:
    # Learners parameters
    window_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 15]) * int(np.sqrt(T))
    window_sizes = np.append(window_sizes, int(2 * np.sqrt(T * np.log(T) / len(settings.phases_duration))))

    # Store the rewards for each experiment for the learners
    swucb_rewards_per_experiment = [[] for _ in window_sizes]

    # Each iteration simulates the learner-environment interaction
    for e in tqdm(range(0, n_experiments)):
        # Define the environment and learners
        environments_swucb = []
        learners_swucb = []
        for window_size in window_sizes:
            environments_swucb.append(NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration))
            learners_swucb.append(SWUCBLearner(settings.prices[category], window_size))

        # Iterate over the number of rounds
        for t in range(0, T):
            for i in range(len(learners_swucb)):
                best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, learners_swucb[i].get_conv_prob(arm) * (environments_swucb[i].prices[category][arm] - environments_swucb[i].other_costs))[0] for arm in range(settings.n_prices)]
                n_clicks_list = np.array([environments_swucb[i].get_n_clicks(category, bid) for bid in best_bids_idx])
                cum_daily_costs_list = np.array([environments_swucb[i].get_cum_daily_costs(category, bid) for bid in best_bids_idx])
                pulled_arm = learners_swucb[i].pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)
                bernoulli_realizations = environments_swucb[i].round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
                reward = environments_swucb[i].get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
                learners_swucb[i].update(pulled_arm, reward, bernoulli_realizations)

        # Store the values of the collected rewards of the learners
        for i, learner in enumerate(learners_swucb):
            swucb_rewards_per_experiment[i].append(learner.collected_rewards)

    # Plot the results
    labels = [r'$1 \cdot \log{T}$', r'$2 \cdot \log{T}$', r'$3 \cdot \log{T}$', r'$4 \cdot \log{T}$', r'$5 \cdot \log{T}$', r'$6 \cdot \log{T}$', r'$7 \cdot \log{T}$', r'$2 \cdot \sqrt{(T \cdot \log{T}) / 3 }$']
    plot_all_algorithms(swucb_rewards_per_experiment, best_rewards, labels)
    for i, label in enumerate(labels):
        plot_single_algorithm(swucb_rewards_per_experiment[i], best_rewards, label, np.arange(0, T, 1))

# CUSUM-UCB analysis
if cusum_ucb:
    # Learners parameters
    # If find better values change them in settings
    M = 50
    eps = 0.1
    h = 0.5 * np.log(T)
    alpha = np.sqrt(np.log(T) / T)

    Ms = [50, 100, 150, 200, 500]
    epss = [0.01, 0.02, 0.05, 0.1, 0.2]
    hs = np.array([0.5, 1, 2, 5, 10]) * np.log(T)
    alphas = np.array([0.1, 0.5, 1, 2, 5]) * np.sqrt(np.log(T) / T)

    # Store the rewards for each experiment for the learners
    cusum_ucb_rewards_per_experiment = [[] for _ in alphas]  # To try other parameters change here

    # Each iteration simulates the learner-environment interaction
    for e in tqdm(range(0, n_experiments)):
        # Define the environment and learners
        environments_cusum_ucb = []
        learners_cusum_ucb = []

        for alpha in alphas:  # To try other parameters change here
            environments_cusum_ucb.append(NonStationaryEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks_cost, settings.bids_to_cum_costs_cost, settings.other_costs, settings.phases_duration))
            learners_cusum_ucb.append(CUSUMUCBLearner(settings.prices[category], M=M, eps=eps, h=h, alpha=alpha))

        # Iterate over the number of rounds
        for t in range(0, T):
            for i in range(len(learners_cusum_ucb)):
                best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, learners_cusum_ucb[i].get_conv_prob(arm) * (environments_cusum_ucb[i].prices[category][arm] - environments_cusum_ucb[i].other_costs))[0] for arm in range(settings.n_prices)]
                n_clicks_list = np.array([environments_cusum_ucb[i].get_n_clicks(category, bid) for bid in best_bids_idx])
                cum_daily_costs_list = np.array([environments_cusum_ucb[i].get_cum_daily_costs(category, bid) for bid in best_bids_idx])
                pulled_arm = learners_cusum_ucb[i].pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)
                bernoulli_realizations = environments_cusum_ucb[i].round_pricing(pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
                reward = environments_cusum_ucb[i].get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
                learners_cusum_ucb[i].update(pulled_arm, reward, bernoulli_realizations)

        # Store the values of the collected rewards of the learners
        for i, learner in enumerate(learners_cusum_ucb):
            cusum_ucb_rewards_per_experiment[i].append(learner.collected_rewards)

    # Plot the results
    labels = ['1', '2', '3', '4', '5']
    plot_all_algorithms(cusum_ucb_rewards_per_experiment, best_rewards, labels)
    for i, label in enumerate(labels):
        plot_single_algorithm(cusum_ucb_rewards_per_experiment[i], best_rewards, label, np.arange(0, T, 1))


"""
Parameters for CUSUM are:
- eps: to have an underestimation of the deviation from the reference point, it's the exploration term of UCB
- M: number of valid samples that are used to compute the reference point
- h: value over which a detection is flagged
- alpha: pure exploration parameter

eps 
The change in the distributions should be greater than 3*eps, anyway eps should be of the order of changes in the 
distributions. With these parameters the smallest change in the distribution is around 0.05, so eps should be at least 
0.01. With eps=0.1 the algorithm performs well, if eps is increased the performance are worse.

M 
If M is too big, so comparable to the length of the horizon, all the observations are used to compute the reference 
points so the algorithm behaves like a normal UCB. If M is very small like 1 or 2 it is suboptimal. M should be big 
enough to have a good estimate of the reference point and such that each arm is played at least M times between each 
breakpoint. In this case since everytime an arm is chosen it receives around 100 clicks, setting M around 50 or 100 is 
enough to have a good estimate of the reference point.

h
A lower threshold makes the algorithm more sensitive, while a higher threshold makes it less sensitive. It should be 
tuned around log(T/number_breakpoints)=2 (from the paper) or should be of the order of logT=6 (notebook Alberto). 
With h=0.5*log(T) the algorithm performs well, increasing the multiplicative factor the performances are worse.

alpha
From the paper it should be tuned around sqrt(log(T/number_breakpoints)*number_breakpoints/T)=0.13 or should be of the
order of sqrt(log(T) / T) (notebook Alberto). I also tried alpha decreasing in time like 1/t, but it is not good because 
over time there is less exploration and so it is more difficult to detect changes. A good alpha seems 
np.sqrt(np.log(T)/T). If alpha is too low the algorithm is suboptimal because doesn't explore, instead if alpha is too
big the algorithm explores too much.

In general all these parameters depend also on the other parameters, e.g. the probabilities.

https://arxiv.org/pdf/1711.03539.pdf it should be the pdf used also during lectures, since the formulas in Alberto's
notebook are the same of the paper by omitting the number of breakpoints, that is usually unknown.
"""

"""
Parameters for Sliding widows UCB:
- windows size: number of valid samples that are used to compute the reference point

windows size
From the paper, remark 9 (pag11) If the horizon T and the growth rate of the number of breakpoints Υ(T) are known in 
advance we can set the windows size to 2 * (upper bound reward) * sqrt(T * log(T) / number of breakpoints) = 53.
The lower is the windows size the higher is the times that the algorithm has to retry all the arms. So, the cumulative 
regret will increase. However, the larger is the window the lower is the sensitivity of the algorithm.
If I don't know all the terms of the previous formula in advance, in the notebook (Alberto) is written to use a window 
proportional to sqrt(T). [1, 2, 3, 4, 5, 6, 7] * int(np.sqrt(365)) = [19,  38,  57,  76,  95, 114, 133]
A good range of values for the window is between 50 and 90, for example 4 * np.sqrt(365) = 76.
Using as multiplicative factor 6 or 7 would be optimal in the case in which the phases have all the same length since  
the window would be similar in length to the phases.
https://arxiv.org/pdf/0805.3415.pdf it should be the original paper of SW-UCB, pag 11
"""

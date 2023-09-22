import numpy as np

from tqdm import tqdm

import settings

from Learners import Clairvoyant, SWUCBLearner, CUSUMUCBLearner
from Environments import NonStationaryEnvironment
from Utilities import plot_all_algorithms, plot_all_algorithms_divided

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

n_experiments = 100

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
    window_sizes = np.array([1, 2, 4, 6, 7]) * int(np.sqrt(T))
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
    labels = [r'$1 \cdot \sqrt{T} = 19$', r'$2 \cdot \sqrt{T} = 38$', r'$4 \cdot \sqrt{T} = 76$', r'$6 \cdot \sqrt{T}$ = 115', r'$7 \cdot \sqrt{T}$ = 134', r'$2 \cdot \sqrt{(T \cdot \log{T}) / 3 } = 54$']
    plot_all_algorithms(swucb_rewards_per_experiment, best_rewards, np.arange(0, T, 1), labels, step_name="sensitivity_analysis_swucb")
    plot_all_algorithms_divided(swucb_rewards_per_experiment, best_rewards, np.arange(0, T, 1), labels, step_name="sensitivity_analysis_swucb")
    #for i, label in enumerate(labels):
    #    plot_single_algorithm(swucb_rewards_per_experiment[i], best_rewards, label, np.arange(0, T, 1))

# CUSUM-UCB analysis
if cusum_ucb:
    # Learners parameters
    M = settings.M
    eps = settings.eps
    h = settings.h
    alpha = settings.alpha

    Ms = [50, 100, 150, 200, 500]
    epss = [0.01, 0.02, 0.05, 0.1, 0.2]
    hs = np.array([0.5, 1, 2, 5, 10]) * np.log(T)
    hs = np.append(hs, np.log(T / len(settings.phases_duration)))
    alphas = np.array([0.1, 0.5, 1, 2, 5]) * np.sqrt(np.log(T) / T)

    # Store the rewards for each experiment for the learners
    cusum_ucb_rewards_per_experiment = [[] for _ in hs]  # To try other parameters change here

    # Each iteration simulates the learner-environment interaction
    for e in tqdm(range(0, n_experiments)):
        # Define the environment and learners
        environments_cusum_ucb = []
        learners_cusum_ucb = []

        for h in hs:  # To try other parameters change here
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
    plot_all_algorithms(cusum_ucb_rewards_per_experiment, best_rewards, np.arange(0, T, 1), labels, step_name="sensitivity_analysis_cusumucb")
    plot_all_algorithms_divided(cusum_ucb_rewards_per_experiment, best_rewards, np.arange(0, T, 1), labels, step_name="sensitivity_analysis_cusumucb")
    #for i, label in enumerate(labels):
    #    plot_single_algorithm(cusum_ucb_rewards_per_experiment[i], best_rewards, label, np.arange(0, T, 1))


"""
Parameters for sliding window UCB:
- windows size: number of valid samples that are used to compute the confidence bounds

windows size
From the paper, remark 9 (pag11) If the horizon T and the growth rate of the number of breakpoints Υ(T) are known in 
advance we can set the windows size to 2 * (upper bound reward) * sqrt(T * log(T) / number of breakpoints) = 54.

If I don't know all the terms of the previous formula in advance, in the notebook (Alberto) is written to use a window 
proportional to sqrt(T). [1, 2, 3, 4, 5, 6, 7] * int(np.sqrt(365)) = [19,  38,  57,  76,  95, 114, 133]

The lower is the window size, the higher is the number of times that the algorithm has to retry all the arms. 
In this case, applying small multiplicative factors, like 1 or 2, decreases the performance because the window is too 
short with respect to the length of the phases and so the learner frequently needs to play suboptimal arms.
While the larger is the window size, the lower is the sensitivity of the algorithm.
Values for the window in the range between 50 and 80, for instance using a factor of 4, are good.
Using as multiplicative factor 6 or 7 would be optimal in this case because the window size would be similar to the 
length of the phases. Applying even higher factors would decrease the performance leading SW-UCB to don’t explore enough.
https://arxiv.org/pdf/0805.3415.pdf it should be the original paper of SW-UCB, pag 11
"""

"""
Parameters for CUSUM are:
- M: number of valid samples that are used to compute the reference point
- eps: to have an underestimation of the deviation from the reference point
- h: value over which a detection is flagged
- alpha: pure exploration parameter

M 
Because the detection test is not run until M samples have been gathered, in general, if M is too large, the majority of
the observations are utilised to compute the reference point, making the algorithm behave similarly to a regular UCB. 
Therefore, with big values of M, it's possible that the changes aren’t detected or take a long time to be identified.
Instead, if M is too small, it is still suboptimal since it leads to inaccurate reference point estimation and more 
frequent change detection, even when there is none.
Each arm should be played at least M times between each breakpoint, and M should be large enough to allow for an 
accurate estimation of the reference point. In this instance, placing M between 50 and 100 is sufficient to get a decent 
estimate of the reference point because each time an arm is chosen, it receives approximately 100 clicks. 
All of the values taken into consideration perform similarly.

eps 
In general, with low values of 𝜀 changes are easily detected even when there aren’t. Whereas as 𝜀 increases the 
sensitivity decreases, so more samples of the new distribution are necessary to flag a change. 
We note that values in the range between 0.01 and 0.1 produce similar performance, while increasing the value of 𝜀 
results in a rise of the cumulative regret, since 𝜀 should be of the order of changes in distributions, which in our 
case are of the order of 0.1.

h
We tried values proportional to np.log(T) (from the paper).
In general, low values of h makes the algorithm more sensitive, while high values make it less sensitive.
With our parameters the best performing h is the one with multiplicative factor 0.5. As the factor increases the 
cumulative regret gets worse since the learner needs more evidence to detect a change in the distribution.
In case the number of breakpoints is known, the paper proposes to use np.log(T / #breakpoints), that would be optimal in 
this case as it is possible to see from the plot.

alpha
We tried values proportional to np.sqrt(np.log(T) / T) (from the paper).
𝛼 is the probability with which a random arm is played in place of the recommended one by UCB.
If 𝛼 is too low, the algorithm is suboptimal since it exploits too much, pulling almost always the suggested arm and 
rarely trying other arms that could became optimal in a non-stationary environment.
Instead, if 𝛼 is too large, the learner explores too much, thus it frequently pulls random arms being able to detect
changes, but not exploiting what has learned.
From the plot we observe that lower values of 𝛼 give lower regret.

In general all these parameters depend also on the other parameters, e.g. the probabilities.

https://arxiv.org/pdf/1711.03539.pdf it should be the pdf used also during lectures, since the formulas in Alberto's
notebook are the same of the paper by omitting the number of breakpoints, that is usually unknown.
"""

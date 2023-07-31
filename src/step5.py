from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUMUCBLearner import CUSUMUCBLearner
from Plots import Plots

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

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.03, 0.04, 0.05, 0.03, 0.01]),  # best arm is 2 (starting from 0)
                 'C2': np.array([0.03, 0.05, 0.03, 0.05, 0.02]),  # best arm is 3
                 'C3': np.array([0.06, 0.07, 0.02, 0.02, 0.01])}  # best arm is 1
bids_to_clicks = {'C1': np.array([100, 2]),  # this curve doesn't change
                  'C2': np.array([100, 2]),
                  'C3': np.array([100, 2])}
bids_to_cum_costs = {'C1': np.array([20, 0.5]),  # this curve doesn't change
                     'C2': np.array([20, 0.5]),
                     'C3': np.array([20, 0.5])}
other_costs = 400
phases_duration = [121, 121, 123]

# Time horizon of the experiment
T = 365
assert np.sum(phases_duration) == T
#window_size = int(T ** 0.5)
window_size = 50

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 200

# Store the rewards for each experiment for the learners
ucb_reward_per_experiment = []
swucb_reward_per_experiment = []
cusum_ucb_reward_per_experiment = []
best_rewards = np.array([])

# Define the environment
env = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)

# Compute the best rewards over the year with the clairvoyant
for phase, phase_len in enumerate(phases_duration):
    # Optimize the problem for each phase
    best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward('C' + str(phase + 1))
    best_rewards = np.append(best_rewards, [best_reward] * phase_len)

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
    #cusum_ucb_learner = CUSUMUCBLearner(prices[category])
    cusum_ucb_learner = CUSUMUCBLearner(prices[category])

    # Iterate over the number of rounds
    for t in range(0, T):
        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        best_bid_idx = clairvoyant.maximize_reward_from_bid(category, ucb_learner.get_conv_prob(pulled_arm) * (
                    env_ucb.prices[category][pulled_arm] - env_ucb.other_costs))[0]
        n_clicks = env_ucb.get_n_clicks(category, best_bid_idx)
        bernoulli_realizations = env_ucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # SW-UCB Learner
        pulled_arm = swucb_learner.pull_arm()
        best_bid_idx = clairvoyant.maximize_reward_from_bid(category, swucb_learner.get_conv_prob(pulled_arm) * (
                env_swucb.prices[category][pulled_arm] - env_swucb.other_costs))[0]
        n_clicks = env_swucb.get_n_clicks(category, best_bid_idx)
        bernoulli_realizations = env_swucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_swucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        swucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # CUSUM-UCB Learner
        pulled_arm = cusum_ucb_learner.pull_arm()
        best_bid_idx = clairvoyant.maximize_reward_from_bid(category, cusum_ucb_learner.get_conv_prob(pulled_arm) * (
                env_cusum_ucb.prices[category][pulled_arm] - env_cusum_ucb.other_costs))[0]
        n_clicks = env_cusum_ucb.get_n_clicks(category, best_bid_idx)
        bernoulli_realizations = env_cusum_ucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_cusum_ucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        cusum_ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    swucb_reward_per_experiment.append(swucb_learner.collected_rewards)
    cusum_ucb_reward_per_experiment.append(cusum_ucb_learner.collected_rewards)

regret_ucb_mean = np.mean(best_rewards - ucb_reward_per_experiment, axis=0)
regret_ucb_std = np.std(best_rewards - ucb_reward_per_experiment, axis=0)
regret_swucb_mean = np.mean(best_rewards - swucb_reward_per_experiment, axis=0)
regret_swucb_std = np.std(best_rewards - swucb_reward_per_experiment, axis=0)
regret_cusum_ucb_mean = np.mean(best_rewards - cusum_ucb_reward_per_experiment, axis=0)
regret_cusum_ucb_std = np.std(best_rewards - cusum_ucb_reward_per_experiment, axis=0)
cumulative_regret_ucb_mean = np.mean(np.cumsum(best_rewards - ucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ucb_std = np.std(np.cumsum(best_rewards - ucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_swucb_mean = np.mean(np.cumsum(best_rewards - swucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_swucb_std = np.std(np.cumsum(best_rewards - swucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_cusum_ucb_mean = np.mean(np.cumsum(best_rewards - cusum_ucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_cusum_ucb_std = np.std(np.cumsum(best_rewards - cusum_ucb_reward_per_experiment, axis=1), axis=0)
reward_ucb_mean = np.mean(ucb_reward_per_experiment, axis=0)
reward_ucb_std = np.std(ucb_reward_per_experiment, axis=0)
reward_swucb_mean = np.mean(swucb_reward_per_experiment, axis=0)
reward_swucb_std = np.std(swucb_reward_per_experiment, axis=0)
reward_cusum_ucb_mean = np.mean(cusum_ucb_reward_per_experiment, axis=0)
reward_cusum_ucb_std = np.std(cusum_ucb_reward_per_experiment, axis=0)
cumulative_reward_ucb_mean = np.mean(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ucb_std = np.std(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_swucb_mean = np.mean(np.cumsum(swucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_swucb_std = np.std(np.cumsum(swucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_cusum_ucb_mean = np.mean(np.cumsum(cusum_ucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_cusum_ucb_std = np.std(np.cumsum(cusum_ucb_reward_per_experiment, axis=1), axis=0)

plots = Plots()
plots.plot_all_algorithms(regret_means=[regret_ucb_mean, regret_swucb_mean, regret_cusum_ucb_mean],
                          cum_regret_means=[cumulative_regret_ucb_mean, cumulative_regret_swucb_mean, cumulative_regret_cusum_ucb_mean],
                          reward_means=[reward_ucb_mean, reward_swucb_mean, reward_cusum_ucb_mean],
                          cum_reward_means=[cumulative_reward_ucb_mean, cumulative_reward_swucb_mean, cumulative_reward_cusum_ucb_mean],
                          best_reward=best_rewards, legend=['UCB', 'SW-UCB', 'CUSUM-UCB'])
"""
plots.plot_single_algorithms(regret_means=[regret_ucb_mean, regret_swucb_mean, regret_cusum_ucb_mean],
                             regret_stds=[regret_ucb_std, regret_swucb_std, regret_cusum_ucb_std],
                             cum_regret_means=[cumulative_regret_ucb_mean, cumulative_regret_swucb_mean, cumulative_regret_cusum_ucb_mean],
                             cum_regret_stds=[cumulative_regret_ucb_std, cumulative_regret_swucb_std, cumulative_regret_cusum_ucb_std],
                             reward_means=[reward_ucb_mean, reward_swucb_mean, reward_cusum_ucb_mean],
                             reward_stds=[reward_ucb_std, reward_swucb_std, reward_cusum_ucb_std],
                             cum_reward_means=[cumulative_reward_ucb_mean, cumulative_reward_swucb_mean, cumulative_reward_cusum_ucb_mean],
                             cum_reward_stds=[cumulative_reward_ucb_std, cumulative_reward_swucb_std, cumulative_reward_cusum_ucb_std],
                             best_reward=best_rewards,
                             legend=['UCB', 'SW-UCB', 'CUSUM-UCB'], x_range=np.arange(0, T, 1))
"""

#TODO controlla gli appunti sul quaderno
#TODO maybe do some plots with different values of parameters to compare them
"""
Parameters for CUSUM are:
- eps, to have an underestimation of the deviation from the reference point, it's the exploration term of UCB
- M, number of valid samples that are used to compute the reference point
- h, value over which a detection is flagged
- alpha, pure exploration parameter

eps 
The change in the rewards should be greater than 3*eps, anyway eps should be of the order of changes in the 
distributions. With these parameters the smallest change in the reward is around 200, so eps should not be greater than 
60. With eps=50 the algorithm performs well, if eps is increased the performance are worse.

M 
If M is too big, so comparable to the length of the horizon, all the observations are used to compute the reference 
points so the algorithm behaves like a normal UCB. If M is very small like 1 or 2 it is suboptimal. M should be big 
enough to have a good estimate of the reference point and such that each arm is played at least M times between each 
breakpoint. So M could be around 5-15.

h
A lower threshold makes the algorithm more sensitive, while a higher threshold makes it less sensitive. It should be 
tuned around log(T/number_breakpoints)=2 (from the paper) or should be of the order of logT=6 (appunti). 
Values in the range 12-60 seem good.

alpha
From the paper it should be tuned around sqrt(log(T/number_breakpoints)*number_breakpoints/T)=0.13. I also tried alpha
decreasing in time like 1/t, but it is not good because over time there is less exploration and so it is more difficult  
to detect changes. A good alpha seems 0.1.

https://arxiv.org/pdf/1711.03539.pdf dovrebbe essere il pdf a cui è stato fatto riferimento anche a lezione
"""

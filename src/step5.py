from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from NonStationaryEnvironment import *
from SWUCB import SWUCBLearner
from CUSUM_UCB_Learner import CUSUM_UCB_Learner

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
n_experiments = 50

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
    # Define the environment
    env_ucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    # Define the learners
    ucb_learner = UCBLearner(prices[category])

    # Define the environment
    env_swucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    swucb_learner = SWUCBLearner(prices[category], window_size)

    env_cusum_ucb = NonStationaryEnvironment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration)
    cusum_ucb_learner = CUSUM_UCB_Learner(prices[category])

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

        # SWUCB Learner
        pulled_arm = swucb_learner.pull_arm()
        best_bid_idx = clairvoyant.maximize_reward_from_bid(category, swucb_learner.get_conv_prob(pulled_arm) * (
                env_swucb.prices[category][pulled_arm] - env_swucb.other_costs))[0]
        n_clicks = env_swucb.get_n_clicks(category, best_bid_idx)
        bernoulli_realizations = env_swucb.round_pricing(pulled_arm, n_clicks=int(np.floor(n_clicks)))
        reward = env_swucb.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        swucb_learner.update(pulled_arm, reward, bernoulli_realizations)

        # CUSUM UCB Learner
        # SWUCB Learner
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

#TODO make functions to plot

# Plot the results, comparison TS-UCB
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
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

axes[0].set_title('Instantaneous regret plot')
axes[0].plot(regret_swucb_mean, 'r')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].plot(regret_cusum_ucb_mean, 'y')
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["SWUCB", "UCB", "CUSUM_UCB"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot')
axes[1].plot(reward_swucb_mean, 'r')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].plot(reward_cusum_ucb_mean, 'y')
axes[1].plot(best_rewards, 'b')
axes[1].legend(["SWUCB", "UCB", "CUSUM_UCB", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot')
axes[2].plot(cumulative_regret_swucb_mean, 'r')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].plot(cumulative_regret_cusum_ucb_mean, 'y')
axes[2].legend(["SWUCB", "UCB", "CUSUM_UCB"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot')
axes[3].plot(cumulative_reward_swucb_mean, 'r')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["SWUCB", "UCB", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# Plot the results for UCB with std
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

axes[0].set_title('Instantaneous regret plot for UCB')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].fill_between(range(0, T), regret_ucb_mean - regret_ucb_std, regret_ucb_mean + regret_ucb_std, color='g', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["UCB mean", "UCB std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot for UCB')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].fill_between(range(0, T), reward_ucb_mean - reward_ucb_std, reward_ucb_mean + reward_ucb_std, color='g', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot for UCB')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].fill_between(range(0, T), cumulative_regret_ucb_mean - cumulative_regret_ucb_std, cumulative_regret_ucb_mean + cumulative_regret_ucb_std, color='g', alpha=0.4)
axes[2].legend(["UCB mean", "UCB std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot for UCB')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].fill_between(range(0, T), cumulative_reward_ucb_mean - cumulative_reward_ucb_std, cumulative_reward_ucb_mean + cumulative_reward_ucb_std, color='g', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["UCB mean", "UCB std", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# Plot the results for SWUCB with std
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

axes[0].set_title('Instantaneous regret plot for SWUCB')
axes[0].plot(regret_swucb_mean, 'r')
axes[0].fill_between(range(0, T), regret_swucb_mean - regret_swucb_std, regret_swucb_mean + regret_swucb_std, color='r', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["SWUCB mean", "SWUCB std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot for SWUCB')
axes[1].plot(reward_swucb_mean, 'r')
axes[1].fill_between(range(0, T), reward_swucb_mean - reward_swucb_std, reward_swucb_mean + reward_swucb_std, color='r', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["SWUCB mean", "SWUCB std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot for SWUCB')
axes[2].plot(cumulative_regret_swucb_mean, 'r')
axes[2].fill_between(range(0, T), cumulative_regret_swucb_mean - cumulative_regret_swucb_std, cumulative_regret_swucb_mean + cumulative_regret_swucb_std, color='r', alpha=0.4)
axes[2].legend(["SWUCB mean", "SWUCB std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot for SWUCB')
axes[3].plot(cumulative_reward_swucb_mean, 'r')
axes[3].fill_between(range(0, T), cumulative_reward_swucb_mean - cumulative_reward_swucb_std, cumulative_reward_swucb_mean + cumulative_reward_swucb_std, color='r', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["SWUCB mean", "SWUCB std", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

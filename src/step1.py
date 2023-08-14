from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from TSReward import *
from collections import Counter
from plots import plot_single_algorithm, plot_all_algorithms
import settings

"""
Simulation for the step 1: learning for pricing 

Consider the case in which all the users belong to class C1. Assume that the curves related to the advertising part of 
the problem are known, while the curve related to the pricing problem is not. Apply the UCB1 and TS algorithms, 
reporting the plots of the average (over a sufficiently large number of runs) value and standard deviation of the 
cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 50

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# To count how many times the best arm is found
ucb_best = []
ts_best = []

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
best_rewards = np.ones((T,)) * best_reward

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    ts_learner = TSRewardLearner(settings.prices[category])
    ucb_learner = UCBLearner(settings.prices[category])

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        # In this scenario the functions of number of clicks and cumulative daily costs are known, so it is possible to
        # optimize the bid given the price. So for each possible price the best bid is computed and then the number of
        # clicks and daily cost. Finally, the arm is pulled.
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, ts_learner.get_conv_prob(arm) * (env.prices[category][arm] - env.other_costs))[0] for arm in range(settings.n_prices)]
        n_clicks_list = np.array([env.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = ts_learner.pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)
        # We take the floor of the number of clicks
        bernoulli_realizations = env.round_pricing(category, pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
        ts_learner.update(pulled_arm, reward, bernoulli_realizations)

        # UCB Learner
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, ucb_learner.get_conv_prob(arm) * (env.prices[category][arm] - env.other_costs))[0] for arm in range(settings.n_prices)]
        n_clicks_list = np.array([env.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = ucb_learner.pull_arm(settings.other_costs, n_clicks_list, cum_daily_costs_list)

        bernoulli_realizations = env.round_pricing(category, pulled_arm, int(np.floor(n_clicks_list[pulled_arm])))
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)

    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

    # Store the best arm found
    ts_best.append(np.argmax(ts_learner.beta_parameters[:, 0] / (ts_learner.beta_parameters[:, 0] + ts_learner.beta_parameters[:, 1])))
    ucb_best.append(np.argmax(ucb_learner.empirical_means + ucb_learner.confidence))

# Print occurrences of best arm in TS
print(Counter(ts_best))
# Print occurrences of best arm in UCB1
print(Counter(ucb_best))

# Plot the results
reward_per_algorithm = [ucb_reward_per_experiment, ts_reward_per_experiment]
labels = ['UCB', 'TS']

plot_all_algorithms(reward_per_algorithm, best_rewards, labels)
for i, label in enumerate(labels):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, label, np.arange(0, T, 1))

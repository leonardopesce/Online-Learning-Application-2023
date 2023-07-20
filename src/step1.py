from tqdm import tqdm
from Clairvoyant import *
from UCB import *
from TSReward import *
from collections import Counter

"""
Simulation for the step 1: learning for pricing 
Consider the case in which all the users belong to class C1. Assume that the curves related to the advertising part of 
the problem are known, while the curve related to the pricing problem is not. Apply the UCB1 and TS algorithms, 
reporting the plots of the average (over a sufficiently large number of runs) value and standard deviation of the 
cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

# Setting the environment parameters
n_prices = 5
arms_values = {'C1': np.array([500, 550, 600, 650, 700]),
               'C2': np.array([500, 550, 600, 650, 700]),
               'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([3, 1, 0.5]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([10, 0.5, 0.5]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 300

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 300

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# To count how many times the best arm is found
ucb_best = []
ts_best = []

# Define the environment
env = Environment(n_prices, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    ts_learner = TSRewardLearner(n_prices)
    ucb_learner = UCBLearner(n_prices)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        n_clicks = env.get_n_clicks(category, best_bid_idx)
        cum_daily_costs = env.get_cum_daily_costs(category, best_bid_idx)
        pulled_arm = ts_learner.pull_arm(arms_values[category], other_costs, n_clicks, cum_daily_costs)
        # TODO check se necessari, cioè se devo ottimizzare i valori
        # devo prendere la conversion prob e quindi il margin per ogni prezzo e trovo la bid migliore per ogni prezzo e
        # conversion_times_margin = env.get_conversion_times_margin(category, pulled_arm, conversion_probability=bernoulli_realization)
        # _, _, reward = clairvoyant.maximize_reward_from_bid(category, conversion_times_margin)
        bernoulli_realizations = np.array([env.round_pricing(pulled_arm, category) for _ in range(0, int(np.floor(n_clicks)))])
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        ts_learner.update(pulled_arm, reward, bernoulli_realizations)

        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        bernoulli_realizations = np.array([env.round_pricing(pulled_arm, category) for _ in range(0, int(np.floor(n_clicks)))])
        #conversion_times_margin = env.get_conversion_times_margin(category, pulled_arm, conversion_probability=bernoulli_realization)
        #_, _, reward = clairvoyant.maximize_reward_from_bid(category, conversion_times_margin)
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        ucb_learner.update(pulled_arm, reward)


    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

    # Store the best arm found
    ucb_best.append(np.argmax(ucb_learner.empirical_means + ucb_learner.confidence))
    ts_best.append(np.argmax(ts_learner.beta_parameters[:, 0] / (ts_learner.beta_parameters[:, 0] + ts_learner.beta_parameters[:, 1])))

# Print occurrences of best arm
print(Counter(ucb_best))
print(Counter(ts_best))

# Plot the results
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
best_rewards = np.ones((T,)) * best_reward
regret_ts_mean = np.mean(best_reward - ts_reward_per_experiment, axis=0)
regret_ts_std = np.std(best_reward - ts_reward_per_experiment, axis=0)
regret_ucb_mean = np.mean(best_reward - ucb_reward_per_experiment, axis=0)
regret_ucb_std = np.std(best_reward - ucb_reward_per_experiment, axis=0)
cumulative_regret_ts_mean = np.mean(np.cumsum(best_reward - ts_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ts_std = np.std(np.cumsum(best_reward - ts_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ucb_mean = np.mean(np.cumsum(best_reward - ucb_reward_per_experiment, axis=1), axis=0)
cumulative_regret_ucb_std = np.std(np.cumsum(best_reward - ucb_reward_per_experiment, axis=1), axis=0)

axes[0].set_title('Instantaneous regret plot')
axes[0].plot(regret_ts_mean, 'r')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].fill_between(range(1, T + 1), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.4)
#axes[0].fill_between(range(1, T + 1), regret_ucb_mean - regret_ucb_std, regret_ucb_mean + regret_ucb_std, color='g', alpha=0.4)
#axes[0].errorbar(range(1, T + 1), regret_ucb_mean, yerr=regret_ucb_std, fmt='-o')
axes[0].legend(["TS", "UCB"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot')
axes[1].plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
axes[1].plot(np.mean(ucb_reward_per_experiment, axis=0), 'g')
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS", "UCB", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].fill_between(range(1, T + 1), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='r', alpha=0.4)
axes[2].legend(["TS", "UCB"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot')
axes[3].plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
axes[3].plot(np.cumsum(np.mean(ucb_reward_per_experiment, axis=0)), 'g')
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["TS", "UCB", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# TODO rimane il mistero del perchè TS vada peggio di UCB nonostante trovi più volte l'arm giusto
# TODO decidere come organizzare i plot fare quelli insieme della media e poi separati con std

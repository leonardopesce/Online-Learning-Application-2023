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
#TODO rimane il mistero del perchè TS vada peggio di UCB nonostante trovi più volte l'arm giusto, ucb c'è qualcosa che non va

# Considered category is C1
category = 'C1'

# Setting the environment parameters
n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
               'C2': np.array([500, 550, 600, 650, 700]),
               'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.03, 0.04, 0.05, 0.03, 0.01]),
                 'C2': np.array([0.05, 0.05, 0.1, 0.2, 0.1]),
                 'C3': np.array([0.1, 0.3, 0.2, 0.05, 0.05])}
bids_to_clicks = {'C1': np.array([100, 2, 0.0]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([20, 0.5, 0.0]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 400

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 200

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

# To count how many times the best arm is found
ucb_best = []
ts_best = []

# Define the environment
env = Environment(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    ts_learner = TSRewardLearner(prices[category])
    ucb_learner = UCBLearner(prices[category])

    # Iterate over the number of rounds
    for t in range(0, T):
        # Simulate the interaction learner-environment
        # TS Learner
        #n_clicks = env.get_n_clicks(category, best_bid_idx)
        #cum_daily_costs = env.get_cum_daily_costs(category, best_bid_idx)
        #pulled_arm = ts_learner.pull_arm(arms_values[category], other_costs, n_clicks, cum_daily_costs)
        # comment this
        # TODO check se necessari, cioè se devo ottimizzare i valori
        # In this scenario the functions of number of clicks and cumulative daily costs are known, so it is possible to
        # optimize the bid given the price. So for each possible price the best bid is computed and then the number of
        # clicks and daily cost. Finally the arm is pulled.
        best_bids_idx = [clairvoyant.maximize_reward_from_bid(category, ts_learner.get_conv_prob(arm) * (env.prices[category][arm] - env.other_costs))[0] for arm in range(n_prices)]
        n_clicks_list = np.array([env.get_n_clicks(category, bid) for bid in best_bids_idx])
        cum_daily_costs_list = np.array([env.get_cum_daily_costs(category, bid) for bid in best_bids_idx])
        pulled_arm = ts_learner.pull_arm(prices[category], other_costs, n_clicks_list, cum_daily_costs_list)
        # conversion_times_margin = env.get_conversion_times_margin(category, pulled_arm, conversion_probability=bernoulli_realization)
        # _, _, reward = clairvoyant.maximize_reward_from_bid(category, conversion_times_margin)
        bernoulli_realizations = np.array([env.round_pricing(pulled_arm, category) for _ in range(0, int(np.floor(n_clicks_list[pulled_arm])))])
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bids_idx[pulled_arm])
        ts_learner.update(pulled_arm, reward, bernoulli_realizations)

        # UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        bernoulli_realizations = np.array([env.round_pricing(pulled_arm, category) for _ in range(0, int(np.floor(n_clicks_list[pulled_arm])))])
        best_bids_idx = clairvoyant.maximize_reward_from_bid(category, ucb_learner.get_conv_prob(pulled_arm) * (env.arms_values[category][pulled_arm] - env.other_costs))[0]
        n_clicks_list = env.get_n_clicks(category, best_bids_idx)
        cum_daily_costs_list = env.get_cum_daily_costs(category, best_bids_idx)
        reward = env.get_reward_from_price(category, pulled_arm, np.mean(bernoulli_realizations), best_bid_idx)
        ucb_learner.update(pulled_arm, reward, bernoulli_realizations)


    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)

    # Store the best arm found
    ucb_best.append(np.argmax(ucb_learner.empirical_means + ucb_learner.confidence))
    ts_best.append(np.argmax(ts_learner.beta_parameters[:, 0] / (ts_learner.beta_parameters[:, 0] + ts_learner.beta_parameters[:, 1])))

# Print occurrences of best arm
print(Counter(ucb_best))
print(Counter(ts_best))

# Plot the results, comparison TS-UCB
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
reward_ts_mean = np.mean(ts_reward_per_experiment, axis=0)
reward_ts_std = np.std(ts_reward_per_experiment, axis=0)
reward_ucb_mean = np.mean(ucb_reward_per_experiment, axis=0)
reward_ucb_std = np.std(ucb_reward_per_experiment, axis=0)
cumulative_reward_ts_mean = np.mean(np.cumsum(ts_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ts_std = np.std(np.cumsum(ts_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ucb_mean = np.mean(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)
cumulative_reward_ucb_std = np.std(np.cumsum(ucb_reward_per_experiment, axis=1), axis=0)

axes[0].set_title('Instantaneous regret plot')
axes[0].plot(regret_ts_mean, 'r')
axes[0].plot(regret_ucb_mean, 'g')
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS", "UCB"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot')
axes[1].plot(reward_ts_mean, 'r')
axes[1].plot(reward_ucb_mean, 'g')
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS", "UCB", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].plot(cumulative_regret_ucb_mean, 'g')
axes[2].legend(["TS", "UCB"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot')
axes[3].plot(cumulative_reward_ts_mean, 'r')
axes[3].plot(cumulative_reward_ucb_mean, 'g')
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["TS", "UCB", "Clairvoyant"])
axes[3].set_xlabel("t")
axes[3].set_ylabel("Cumulative reward")
plt.show()

# Plot the results for TS with std
_, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

axes[0].set_title('Instantaneous regret plot for TS')
axes[0].plot(regret_ts_mean, 'r')
axes[0].fill_between(range(0, T), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.4)
axes[0].axhline(y=0, color='b', linestyle='--')
axes[0].legend(["TS mean", "TS std"])
axes[0].set_xlabel("t")
axes[0].set_ylabel("Instantaneous regret")

axes[1].set_title('Instantaneous reward plot for TS')
axes[1].plot(reward_ts_mean, 'r')
axes[1].fill_between(range(0, T), reward_ts_mean - reward_ts_std, reward_ts_mean + reward_ts_std, color='r', alpha=0.4)
axes[1].plot(best_rewards, 'b')
axes[1].legend(["TS mean", "TS std", "Clairvoyant"])
axes[1].set_xlabel("t")
axes[1].set_ylabel("Instantaneous reward")

axes[2].set_title('Cumulative regret plot for TS')
axes[2].plot(cumulative_regret_ts_mean, 'r')
axes[2].fill_between(range(0, T), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='r', alpha=0.4)
axes[2].legend(["TS mean", "TS std"])
axes[2].set_xlabel("t")
axes[2].set_ylabel("Cumulative regret")

axes[3].set_title('Cumulative reward plot for TS')
axes[3].plot(cumulative_reward_ts_mean, 'r')
axes[3].fill_between(range(0, T), cumulative_reward_ts_mean - cumulative_reward_ts_std, cumulative_reward_ts_mean + cumulative_reward_ts_std, color='r', alpha=0.4)
axes[3].plot(np.cumsum(best_rewards), 'b')
axes[3].legend(["TS mean", "TS std", "Clairvoyant"])
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

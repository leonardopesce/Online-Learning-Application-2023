import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from GPTS_Learner import *
from tqdm import tqdm
from Clairvoyant import Clairvoyant

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
bids_to_clicks = {'C1': np.array([1, 1, 0.5]),
                  'C2': np.array([2, 2, 0.5]),
                  'C3': np.array([3, 3, 0.5])}
bids_to_cum_costs = {'C1': np.array([100, 0.5, 0.5]),
                     'C2': np.array([2, 2, 0.5]),
                     'C3': np.array([3, 3, 0.5])}
other_costs = 200

# Bids setup
n_bids = 100
min_bid = 0.5
max_bid = 20.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma = 2

# Time horizon and experiments
T = 365
n_experiments = 1
gpts_rewards_per_experiment = []
gpts_clicks_per_experiment = []
gpts_mean_clicks_per_experiment = []
gpts_sigmas_clicks_per_experiment = []
gpts_cum_costs_per_experiment = []
gpts_mean_cum_costs_per_experiment = []
gpts_sigmas_cum_costs_per_experiment = []
gpts_pulled_bids_per_experiment = []

# Define the environment
env = Environment(n_prices, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)

for e in tqdm(range(0, n_experiments)):
    gpts_learner = GPTS_Learner(arms=bids)

    for t in range(0, T):
        # GP Thompson Sampling
        price_idx, price, prob_margin = clairvoyant.maximize_reward_from_price(category)
        pulled_arm = gpts_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(pulled_arm, category)

        reward = env.get_reward(category=category, price_idx=price_idx, n_clicks=n_clicks, cum_daily_costs=costs)
        
        # Here we update the internal state of the learner passing it the reward,
        # the number of clicks and the costs sampled from the environment.
        gpts_learner.update(pulled_arm, (reward, n_clicks, costs))
    # gpts_learner.plot_clicks()
    # gpts_learner.plot_costs()
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpts_clicks_per_experiment.append(gpts_learner.collected_clicks)
    gpts_mean_clicks_per_experiment.append(gpts_learner.means_clicks)
    gpts_sigmas_clicks_per_experiment.append(gpts_learner.sigmas_clicks)
    gpts_cum_costs_per_experiment.append(gpts_learner.collected_costs)
    gpts_mean_cum_costs_per_experiment.append(gpts_learner.means_costs)
    gpts_sigmas_cum_costs_per_experiment.append(gpts_learner.sigmas_costs)
    gpts_pulled_bids_per_experiment.append(gpts_learner.pulled_bids)

def plot_adv_curves():
    plt.figure(0)
    #plt.scatter(gpts_pulled_bids_per_experiment, np.mean(np.array(gpts_clicks_per_experiment), axis=0), color='r', label='prova mike')
    plt.plot(bids, np.mean(np.array(gpts_mean_clicks_per_experiment), axis=0), color='r', label='mean clicks')
    plt.fill_between(bids, np.mean(np.array(gpts_mean_clicks_per_experiment), axis=0) - np.mean(np.array(gpts_sigmas_clicks_per_experiment), axis=0), np.mean(np.array(gpts_mean_clicks_per_experiment), axis=0) + np.mean(np.array(gpts_sigmas_clicks_per_experiment), axis=0), alpha=0.2, color='r')

    plt.figure(1)
    plt.plot(bids, np.mean(np.array(gpts_mean_cum_costs_per_experiment), axis=0), color='b', label='mean costs')
    plt.fill_between(bids, np.mean(np.array(gpts_mean_cum_costs_per_experiment), axis=0) - np.mean(np.array(gpts_sigmas_cum_costs_per_experiment), axis=0),
                     np.mean(np.array(gpts_mean_cum_costs_per_experiment), axis=0) + np.mean(np.array(gpts_sigmas_cum_costs_per_experiment), axis=0), alpha=0.2, color='b')
    plt.legend()
    plt.show()

def plot_instantaneous_regret() -> None:
    regret_ts_mean = np.mean(best_reward - np.array(gpts_rewards_per_experiment), axis=0)
    regret_ts_std = np.std(best_reward - gpts_rewards_per_experiment, axis=0)

    plt.figure(2)
    plt.plot(regret_ts_mean, 'r', label='Instantaneous Regret')
    plt.fill_between(range(0, T), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.2)
    plt.show()

def plot_cumulative_regret() -> None:
    cumulative_regret_ts_mean = np.mean(np.cumsum(best_reward - gpts_rewards_per_experiment, axis=1), axis=0)
    cumulative_regret_ts_std = np.std(np.cumsum(best_reward - gpts_rewards_per_experiment, axis=1), axis=0)

    plt.figure(3)
    plt.plot(cumulative_regret_ts_mean, 'b', label='Cumulative Regret')
    plt.fill_between(range(0,T), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='b', alpha=0.2)
    plt.show()

#opt = np.max(env.means)
#plt.figure(0)
#plt.xlabel('Bids')
#plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'g')
#plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)))
# plt.plot(np.mean(gpts_clicks_per_experiment, axis=0), color='g')
# plt.plot(np.mean(gpts_cum_costs_per_experiment, axis=0), color='b')
#plt.show()

plot_adv_curves()
#plot_instantaneous_regret()
plot_cumulative_regret()

# TODO tutti i plot da fare
# TODO capire se ha senso plottare i punti delle curve
# TODO GP-UCB
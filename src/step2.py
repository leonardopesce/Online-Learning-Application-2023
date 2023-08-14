import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from GPTS_Learner import *
from GPUCB_Learner import *
from tqdm import tqdm
from Clairvoyant import Clairvoyant
import settings


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Considered category is C1
category = 'C1'

bids = np.linspace(settings.min_bid, settings.max_bid, settings.n_bids)
# Time horizon and experiments
T = 365
n_experiments = 20
gpts_rewards_per_experiment = []
gpts_clicks_per_experiment = []
gpts_mean_clicks_per_experiment = []
gpts_lower_bounds_clicks_per_experiment = []
gpts_upper_bounds_clicks_per_experiment = []
gpts_cum_costs_per_experiment = []
gpts_mean_cum_costs_per_experiment = []
gpts_lower_bounds_costs_per_experiment = []
gpts_upper_bounds_costs_per_experiment = []

gpucb_rewards_per_experiment = []
gpucb_clicks_per_experiment = []
gpucb_mean_clicks_per_experiment = []
gpucb_lower_bounds_clicks_per_experiment = []
gpucb_upper_bounds_clicks_per_experiment = []
gpucb_cum_costs_per_experiment = []
gpucb_mean_cum_costs_per_experiment = []
gpucb_lower_bounds_costs_per_experiment = []
gpucb_upper_bounds_costs_per_experiment = []

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
best_rewards = np.ones((T,)) * best_reward

for e in tqdm(range(0, n_experiments)):
    gpts_learner = GPTS_Learner(arms=bids)
    gpucb_learner = GPUCB_Learner(arms_values=bids)
    #TODO sarebbe da creare l'environment, anche in step3

    for t in range(0, T):
        price_idx, price, prob_margin = clairvoyant.maximize_reward_from_price(category)

        # GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(category, pulled_arm)
        reward_gpts = env.get_reward(category=category, price_idx=price_idx, conversion_prob=settings.probabilities[category][price_idx], n_clicks=n_clicks, cum_daily_costs=costs)

        # GP UCB
        pulled_arm = gpucb_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(category, pulled_arm)
        reward_gpucb = env.get_reward(category=category, price_idx=price_idx, conversion_prob=settings.probabilities[category][price_idx], n_clicks=n_clicks, cum_daily_costs=costs)
        
        # Here we update the internal state of the learner passing it the reward,
        # the number of clicks and the costs sampled from the environment.
        gpts_learner.update(pulled_arm, (reward_gpts, n_clicks, costs))
        gpucb_learner.update(pulled_arm, (reward_gpucb, n_clicks, costs))
    # gpts_learner.plot_clicks()
    # gpts_learner.plot_costs()
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpts_clicks_per_experiment.append(gpts_learner.collected_clicks)
    gpts_mean_clicks_per_experiment.append(gpts_learner.means_clicks)
    gpts_lower_bounds_clicks_per_experiment.append(gpts_learner.lower_bounds_clicks)
    gpts_upper_bounds_clicks_per_experiment.append(gpts_learner.upper_bounds_clicks)
    gpts_cum_costs_per_experiment.append(gpts_learner.collected_costs)
    gpts_mean_cum_costs_per_experiment.append(gpts_learner.means_costs)
    gpts_lower_bounds_costs_per_experiment.append(gpts_learner.lower_bounds_costs)
    gpts_upper_bounds_costs_per_experiment.append(gpts_learner.upper_bounds_costs)

    gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)
    gpucb_clicks_per_experiment.append(gpucb_learner.collected_clicks)
    gpucb_mean_clicks_per_experiment.append(gpucb_learner.empirical_means_clicks)
    gpucb_lower_bounds_clicks_per_experiment.append(gpucb_learner.lower_bounds_clicks)
    gpucb_upper_bounds_clicks_per_experiment.append(gpucb_learner.upper_bounds_clicks)
    gpucb_cum_costs_per_experiment.append(gpucb_learner.collected_costs)
    gpucb_mean_cum_costs_per_experiment.append(gpucb_learner.empirical_means_costs)
    gpucb_lower_bounds_costs_per_experiment.append(gpucb_learner.lower_bounds_costs)
    gpucb_upper_bounds_costs_per_experiment.append(gpucb_learner.upper_bounds_costs)

def plot_adv_curves():
    #Plot Clicks' curve
    plt.figure(0)
    plt.title('Clicks GP')
    #Plot GP-TS
    plt.plot(bids, np.mean(np.array(gpts_mean_clicks_per_experiment), axis=0), color='r', label='GP-TS')
    plt.fill_between(bids, np.mean(np.array(gpts_lower_bounds_clicks_per_experiment), axis=0), np.mean(np.array(gpts_upper_bounds_clicks_per_experiment), axis=0), alpha=0.2, color='r')
    #Plot GP-UCB
    plt.plot(bids, np.mean(np.array(gpucb_mean_clicks_per_experiment), axis=0), color='b', label='GP-UCB')
    plt.fill_between(bids, np.mean(np.array(gpucb_lower_bounds_clicks_per_experiment), axis=0), np.mean(np.array(gpucb_upper_bounds_clicks_per_experiment), axis=0), alpha=0.2, color='b')
    plt.legend()
    plt.show()

    #Plot Costs' curve
    plt.figure(1)
    plt.title('Costs GP')
    #Plot GP-TS
    plt.plot(bids, np.mean(np.array(gpts_mean_cum_costs_per_experiment), axis=0), color='r', label='GP-TS')
    plt.fill_between(bids, np.mean(np.array(gpts_lower_bounds_costs_per_experiment), axis=0), np.mean(np.array(gpts_upper_bounds_costs_per_experiment), axis=0), alpha=0.2, color='r')
    #Plot GP-UCB
    plt.plot(bids, np.mean(np.array(gpucb_mean_cum_costs_per_experiment), axis=0), color='b', label='GP-UCB')
    plt.fill_between(bids, np.mean(np.array(gpucb_lower_bounds_costs_per_experiment), axis=0), np.mean(np.array(gpucb_upper_bounds_costs_per_experiment), axis=0), alpha=0.2, color='b')

    plt.legend()
    plt.show()

def plot_instantaneous_regret() -> None:
    regret_ts_mean = np.mean(best_reward - np.array(gpts_rewards_per_experiment), axis=0)
    regret_ts_std = np.std(best_reward - gpts_rewards_per_experiment, axis=0)
    regret_ucb_mean = np.mean(best_reward - np.array(gpucb_rewards_per_experiment), axis=0)
    regret_ucb_std = np.std(best_reward - gpucb_rewards_per_experiment, axis=0)

    plt.figure(2)
    plt.title('Instantaneous Regret')
    #Plot GP-TS
    plt.plot(regret_ts_mean, 'r', label='GP-TS')
    plt.fill_between(range(0, T), regret_ts_mean - regret_ts_std, regret_ts_mean + regret_ts_std, color='r', alpha=0.2)
    #Plot GP-UCB
    plt.plot(regret_ucb_mean, 'b', label='GP-UCB')
    plt.fill_between(range(0, T), regret_ucb_mean - regret_ucb_std, regret_ucb_mean + regret_ucb_std, color='b', alpha=0.2)

    plt.legend()
    plt.show()

def plot_cumulative_regret() -> None:
    cumulative_regret_ts_mean = np.mean(np.cumsum(best_reward - gpts_rewards_per_experiment, axis=1), axis=0)
    cumulative_regret_ts_std = np.std(np.cumsum(best_reward - gpts_rewards_per_experiment, axis=1), axis=0)
    cumulative_regret_ucb_mean = np.mean(np.cumsum(best_reward - gpucb_rewards_per_experiment, axis=1), axis=0)
    cumulative_regret_ucb_std = np.std(np.cumsum(best_reward - gpucb_rewards_per_experiment, axis=1), axis=0)

    plt.figure(3)
    plt.title('Cumulative Regret')
    #Plot GP-TS
    plt.plot(cumulative_regret_ts_mean, 'r', label='GP-TS')
    plt.fill_between(range(0,T), cumulative_regret_ts_mean - cumulative_regret_ts_std, cumulative_regret_ts_mean + cumulative_regret_ts_std, color='r', alpha=0.2)
    #Plot GP-UCB
    plt.plot(cumulative_regret_ucb_mean, 'b', label='GP-UCB')
    plt.fill_between(range(0, T), cumulative_regret_ucb_mean - cumulative_regret_ucb_std, cumulative_regret_ucb_mean + cumulative_regret_ucb_std, color='b', alpha=0.2)

    plt.legend()
    plt.show()

def plot_instantaneous_reward() -> None:
    plt.figure(3)
    plt.title('Instantaneous Reward')
    plt.plot(np.mean(np.array(gpts_rewards_per_experiment), axis=0), 'r', label='GP-TS')
    plt.plot(np.mean(np.array(gpucb_rewards_per_experiment), axis=0), 'g', label='GP-UCB')
    plt.plot(best_rewards, 'b', label='Best Reward')
    plt.legend()
    plt.show()

def plot_cumulative_reward() -> None:
    plt.figure(4)
    plt.title('Cumulative Reward')
    plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r', label='GP-TS')
    plt.plot(np.cumsum(np.mean(gpucb_rewards_per_experiment, axis=0)), 'g', label='GP-UCB')
    plt.plot(np.cumsum(best_rewards), 'b', label='Best Reward')
    plt.legend()
    plt.show()

plot_adv_curves()
plot_instantaneous_regret()
plot_cumulative_regret()
plot_instantaneous_reward()
plot_cumulative_reward()
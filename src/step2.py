import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from GPTS_Learner import *
from tqdm import tqdm

# Considered category is C1
category = 'C1'

# Setting the environment parameters
n_arms = 5
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
T = 10
n_experiments = 2
gpts_rewards_per_experiment = []
gpts_clicks_per_experiment = []
gpts_mean_clicks_per_experiment = []
gpts_sigmas_clicks_per_experiment = []
gpts_cum_costs_per_experiment = []
gpts_mean_cum_costs_per_experiment = []
gpts_sigmas_cum_costs_per_experiment = []
gpts_pulled_bids_per_experiment = []




def maximize_reward_from_price(cat, environ):
    values = np.array([])
    for idx, price in enumerate(environ.arms_values[cat]):
        values = np.append(values, environ.probabilities[cat][idx] * (price - environ.other_costs))

    best_price_idx = np.random.choice(np.where(values == values.max())[0])
    return best_price_idx, values[best_price_idx]

for e in range(0, n_experiments):
    env = Environment(n_arms, arms_values, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
    gpts_learner = GPTS_Learner(n_arms=n_bids, arms=bids)

    for t in tqdm(range(0, T)):
        # GP Thompson Sampling
        price_idx, prob_margin = maximize_reward_from_price(category, env)
        pulled_arm = gpts_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(pulled_arm, category)

        reward = env.reward(category=category, price_idx=price_idx, n_clicks=n_clicks, cum_daily_costs=costs)
        
        # Here we update the internal state of the learner passing it the reward,
        # the number of clicks and the costs sampled from the environment.
        gpts_learner.update(pulled_arm, (reward, n_clicks, costs))
        '''
        # Plotting the collected clicks after each round.
        if t == T-1:
            gpts_learner.plot_clicks()
            gpts_learner.plot_costs()'''

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpts_clicks_per_experiment.append(gpts_learner.collected_clicks)
    gpts_mean_clicks_per_experiment.append(gpts_learner.means_clicks)
    gpts_sigmas_clicks_per_experiment.append(gpts_learner.sigmas_clicks)
    gpts_cum_costs_per_experiment.append(gpts_learner.collected_costs)
    gpts_mean_cum_costs_per_experiment.append(gpts_learner.means_costs)
    gpts_sigmas_cum_costs_per_experiment.append(gpts_learner.sigmas_costs)
    gpts_pulled_bids_per_experiment.append(gpts_learner.pulled_bids)

def plot():
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

#opt = np.max(env.means)
#plt.figure(0)
#plt.xlabel('Bids')
#plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'g')
#plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)))
# plt.plot(np.mean(gpts_clicks_per_experiment, axis=0), color='g')
# plt.plot(np.mean(gpts_cum_costs_per_experiment, axis=0), color='b')
#plt.show()

plot()
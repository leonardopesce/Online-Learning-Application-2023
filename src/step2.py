from Environment import *
from GPTS_Learner import *
from GPUCB_Learner import *
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from plots import plot_single_algorithm, plot_all_algorithms, plot_clicks_curve, plot_costs_curve
import settings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Step 2: Learning for advertising

Consider the case in which all the users belong to class C1. Assume that the curve related to the pricing problem is 
known while the curves related to the advertising problems are not. Apply the GP-UCB and GP-TS algorithms when using GPs 
to model the two advertising curves, reporting the plots of the average (over a sufficiently large number of runs) value 
and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
"""

# Considered category is C1
category = 'C1'

bids = np.linspace(settings.min_bid, settings.max_bid, settings.n_bids)
# Time horizon and experiments
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 10

algorithms = ['UCB', 'TS']
# To store the learners to plot the advertising curves
gp_learners = {algorithm: [] for algorithm in algorithms}

# Store the rewards for each experiment for the learners
ts_reward_per_experiment = []
ucb_reward_per_experiment = []

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
    # GP-TS learner
    gpts_learner = GPTS_Learner(arms=bids)
    # GP-UCB learner
    gpucb_learner = GPUCB_Learner(arms_values=bids)

    # Iterate over the number of rounds
    for t in range(0, T):
        price_idx, price, prob_margin = clairvoyant.maximize_reward_from_price(category)
        # Simulate the interaction learner-environment
        # GP-TS learner
        pulled_arm = gpts_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(category, pulled_arm)
        reward_gpts = env.get_reward(category=category, price_idx=price_idx,
                                     conversion_prob=settings.probabilities[category][price_idx], n_clicks=n_clicks,
                                     cum_daily_costs=costs)

        # GP-UCB learner
        pulled_arm = gpucb_learner.pull_arm_GPs(prob_margin)
        n_clicks, costs = env.round_advertising(category, pulled_arm)
        reward_gpucb = env.get_reward(category=category, price_idx=price_idx,
                                      conversion_prob=settings.probabilities[category][price_idx], n_clicks=n_clicks,
                                      cum_daily_costs=costs)

        # Here we update the internal state of the learner passing it the reward,
        # the number of clicks and the costs sampled from the environment.
        gpts_learner.update(pulled_arm, (reward_gpts, n_clicks, costs))
        gpucb_learner.update(pulled_arm, (reward_gpucb, n_clicks, costs))

    # Store the values of the collected rewards of the learners
    ts_reward_per_experiment.append(gpts_learner.collected_rewards)
    ucb_reward_per_experiment.append(gpucb_learner.collected_rewards)

    # Store the learners
    gp_learners['TS'].append(gpts_learner)
    gp_learners['UCB'].append(gpucb_learner)

    # gpts_learner.plot_clicks()
    # gpts_learner.plot_costs()

# Plot the results
reward_per_algorithm = [ucb_reward_per_experiment, ts_reward_per_experiment]
plot_clicks_curve(bids, gp_learners, algorithms)
plot_costs_curve(bids, gp_learners, algorithms)
plot_all_algorithms(reward_per_algorithm, best_rewards, algorithms)
for i, label in enumerate(algorithms):
    plot_single_algorithm(reward_per_algorithm[i], best_rewards, label, np.arange(0, T, 1))

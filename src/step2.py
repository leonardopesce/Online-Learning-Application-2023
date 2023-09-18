from Environment import *
from GPTS_Learner import *
from GPUCB_Learner import *
from tqdm import tqdm
from Clairvoyant import Clairvoyant
from plots import plot_single_algorithm, plot_all_algorithms, plot_clicks_curve, plot_costs_curve, plot_all_algorithms_divided
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
n_experiments = 100

algorithms = ['UCB', 'TS']

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
environments = dict()
rewards = {algorithm: [] for algorithm in algorithms}
best_rewards = np.array([])

# To store the learners to plot the advertising curves
gp_learners = {algorithm: [] for algorithm in algorithms}

# Define the environment
env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
# Define the clairvoyant
clairvoyant = Clairvoyant(env)
# Optimize the problem
best_price_idx, best_price, best_bid_idx, best_bid, best_reward = clairvoyant.maximize_reward(category)
best_rewards = np.append(best_rewards, np.ones((T,)) * best_reward)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the environments
    environments = {algorithm: Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs) for algorithm in algorithms}

    # Define the GP learners
    learners['UCB'] = GPUCB_Learner(arms_values=bids, sklearn=False)
    learners['TS'] = GPTS_Learner(arms=bids, sklearn=False)

    # Iterate over the number of rounds
    for t in range(0, T):
        price_idx, price, prob_margin = clairvoyant.maximize_reward_from_price(category)
        for algorithm in algorithms:
            pulled_arm = learners[algorithm].pull_arm_GPs(prob_margin)
            n_clicks, costs = environments[algorithm].round_advertising(category, pulled_arm)
            reward = environments[algorithm].get_reward(category=category, price_idx=price_idx, conversion_prob=settings.probabilities[category][price_idx], n_clicks=n_clicks, cum_daily_costs=costs)
            # Update the internal state of the learner passing it the reward, the number of clicks and the costs sampled
            # from the environment.
            learners[algorithm].update(pulled_arm, (reward, n_clicks, costs))

    # Store the values of the collected rewards of the learners and the learners
    for algorithm in algorithms:
        rewards[algorithm].append(learners[algorithm].collected_rewards)
        gp_learners[algorithm].append(learners[algorithm])
        # learners[algorithm].plot_clicks()
        # learners[algorithm].plot_costs()

# Plot the results
reward_per_algorithm = [rewards[algorithm] for algorithm in algorithms]
plot_clicks_curve(bids, gp_learners, algorithms, original=env.get_clicks_curve(bids, category), step_name="step2")
plot_costs_curve(bids, gp_learners, algorithms, original=env.get_costs_curve(bids, category), step_name="step2")
plot_all_algorithms(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, "step2")
plot_all_algorithms_divided(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, "step2")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

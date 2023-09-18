from tqdm import tqdm
from Clairvoyant import Clairvoyant
import numpy as np
from MultiContextEnvironment import MultiContextEnvironment
from ContextGeneratorLearner import ContextGeneratorLearner
import settings
from plots import plot_single_algorithm, plot_all_algorithms, plot_all_algorithms_divided

"""
Consider the case in which there are three classes of users (C1, C2, and C3), 
and no information about the advertising and pricing curves is known beforehand. 

Consider two scenarios. 
In the first one, the structure of the contexts is known beforehand. 
Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves, 
reporting the plots with the average (over a sufficiently large number of runs) value and standard deviation 
of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward. 

In the second scenario, the structure of the contexts is not known beforehand and needs to be learnt from data. 
Important remark: the learner does not know how many contexts there are, 
while it can only observe the features and data associated with the features. 
Apply the GP-UCB and GP-TS algorithms when using GPs to model the two advertising curves paired with 
a context generation algorithm, reporting the plots with the average (over a sufficiently large number of runs) 
value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, 
and instantaneous reward. Apply the context generation algorithms every two weeks of the simulation. 
Compare the performance of the two algorithms --- the one used in the first scenario with 
the one used in the second scenario. Furthermore, in the second scenario, 
run the GP-UCB and GP-TS algorithms without context generation, and therefore forcing the context to be only one 
for the entire time horizon, and compare their performance with the performance of the previous algorithms used 
for the second scenario.
"""

# Considered categories
categories = ['C1', 'C2', 'C3']

# Considered features and values (binary)
feature_names = ['age', 'sex']
feature_values = {'age': [0, 1], 'sex': [0, 1]}
# age: 0 -> young, 1 -> old; sex: 0 -> not clicked, 1 -> clicked
feature_values_to_categories = {(0, 0): 'C3', (0, 1): 'C1', (1, 0): 'C3', (1, 1): 'C2'}
probability_feature_values_in_categories = {'C1': {(0, 1): 1}, 'C2': {(1, 1): 1}, 'C3': {(0, 0): 0.5, (1, 0): 0.5}}

# Bids setup
bids = np.linspace(settings.min_bid, settings.max_bid, settings.n_bids)

# Time horizon of the experiment
T = 365

# Since the reward functions are stochastic to better visualize the results and remove the noise
# we have to perform a sufficiently large number experiments
n_experiments = 5
time_between_context_generation = 14

algorithms = ['UCB', 'TS']

# To store the learners, environments and rewards for each experiment for the learners
learners = dict()
rewards_per_algorithm = {algorithm: [] for algorithm in algorithms}
best_rewards = np.array([])

"""
# Store some features for each experiment for the learners
gpts_clicks_per_experiment = 0
gpts_mean_clicks_per_experiment = 0
gpts_sigmas_clicks_per_experiment = 0
gpts_cum_costs_per_experiment = 0
gpts_mean_cum_costs_per_experiment = 0
gpts_sigmas_cum_costs_per_experiment = 0
gpts_pulled_bids_per_experiment = 0
"""

# To evaluate which are the most played prices and bids
#ts_best_price, ts_best_bid, ucb_best_price, ucb_best_bid = [], [], [], []
#TS, UCB = 0, 1

# Define the environment
env = MultiContextEnvironment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs,
                              categories, feature_names, feature_values, feature_values_to_categories,
                              probability_feature_values_in_categories)

env.plot_rewards(categories=['C1', 'C2', 'C3'], plot_aggregate_model=True)
env.plot_whole_advertising_model()

# Define the clairvoyant
clairvoyant = Clairvoyant(env)

best_reward = 0
for category in categories:
    _, _, _, _, best_reward_category = clairvoyant.maximize_reward(category)
    best_reward += best_reward_category
best_rewards = np.append(best_rewards, np.ones((T,)) * best_reward)

# Each iteration simulates the learner-environment interaction
for e in tqdm(range(0, n_experiments)):
    # Define the learners
    for algorithm in algorithms:
        learners[algorithm] = ContextGeneratorLearner(settings.prices['C1'], env.bids, env.feature_name, env.feature_values,
                                                 time_between_context_generation, algorithm, settings.other_costs)

    # Iterate over the number of rounds
    for t in range(0, T):
        # Apply the context generation algorithm offline every 2 weeks (i.e. t multiple of 14).
        if t % time_between_context_generation == 0 and t != 0:
            print("--------------------------------------------")
            print("IT'S TIME TO UPDATE THE CONTEXT")
            print(f"TIME: {t}")
            for context_learner in learners.values():
                print("--------------------------------------------")
                context_learner.update_context()

        # Iterate over TS and UCB
        for context_learner in learners.values():
            # Pull all the arm of the context generator
            context_price_bid_learners = context_learner.pull_arm(settings.other_costs)
            # Create variable to update the context learner
            features_list, pulled_price_list, bernoulli_realizations_list, pulled_bid_list = [], [], [], []
            clicks_given_bid_list, cost_given_bid_list, rewards = [], [], []

            # Iterate over the generated contexts
            for context, price_idx, bid_idx in context_price_bid_learners:
                feature_list, bernoulli_realizations, n_clicks, cum_daily_cost = env.round(price_idx, bid_idx, context)

                # TODO: may be still based on category and not on features
                reward = []
                for i, feature in enumerate(feature_list):
                    reward.append(env.get_reward(feature, price_idx, float(np.mean(bernoulli_realizations[i])), n_clicks[i], cum_daily_cost[i]))

                # Prepare data for update of context learner
                features_list.append(feature_list)
                bernoulli_realizations_list.append(bernoulli_realizations)
                pulled_price_list.append(price_idx)
                pulled_bid_list.append(bid_idx)
                clicks_given_bid_list.append(n_clicks)
                cost_given_bid_list.append(cum_daily_cost)
                rewards.append(reward)

            context_learner.update(pulled_price_list=pulled_price_list, bernoulli_realizations_list=bernoulli_realizations_list,
                       features_list=features_list, pulled_bid_list=pulled_bid_list,
                       clicks_given_bid_list=clicks_given_bid_list, cost_given_bid_list=cost_given_bid_list,
                       rewards=rewards)


    # Store the most played prices and bids by TS
    #ts_best_price.append(Counter(context_learners_type[TS].get_pulled_prices()).most_common(1)[0])
    #ts_best_bid.append(Counter(context_learners_type[TS].get_pulled_bids()).most_common(1)[0])

    # Store the most played prices and bids by UCB1
    #ucb_best_price.append(Counter(context_learners_type[UCB].get_pulled_prices()).most_common(1)[0])
    #ucb_best_bid.append(Counter(context_learners_type[UCB].get_pulled_bids()).most_common(1)[0])

    # Store the values of the collected rewards of the learners
    for algorithm in algorithms:
        rewards_per_algorithm[algorithm].append(learners[algorithm].get_collective_reward())

    """gpts_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.collected_clicks)
    gpts_mean_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.means_clicks)
    gpts_sigmas_clicks_per_experiment[category].append(ts_learner[category].GPTS_advertising.sigmas_clicks)
    gpts_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.collected_costs)
    gpts_mean_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.means_costs)
    gpts_sigmas_cum_costs_per_experiment[category].append(ts_learner[category].GPTS_advertising.sigmas_costs)
    gpts_pulled_bids_per_experiment[category].append(ts_learner[category].GPTS_advertising.pulled_bids)"""

# Print occurrences of best arm in TS
#print(Counter(ts_best_price))
#print(Counter(ts_best_bid))
# Print occurrences of best arm in UCB1
#print(Counter(ucb_best_price))
#print(Counter(ucb_best_bid))

# Plot the results
reward_per_algorithm = [rewards_per_algorithm[algorithm] for algorithm in algorithms]
plot_all_algorithms(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_2")
plot_all_algorithms_divided(reward_per_algorithm, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_2")
#for i, algorithm in enumerate(algorithms):
#    plot_single_algorithm(reward_per_algorithm[i], best_rewards, algorithm, np.arange(0, T, 1))

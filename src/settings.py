import numpy as np

"""
In this Python file, all settings parameters are defined, and each step takes the necessary parameters.
"""

T = 365

n_prices = 5
prices = {'C1': np.array([500, 550, 600, 650, 700]),
          'C2': np.array([500, 550, 600, 650, 700]),
          'C3': np.array([500, 550, 600, 650, 700])}
probabilities = {'C1': np.array([0.10, 0.12, 0.20, 0.04, 0.03]),  # best arm is 2 (starting from 0), young who searched technology terms
                 'C2': np.array([0.03, 0.04, 0.10, 0.12, 0.20]),  # best arm is 4, old who searched technology terms
                 'C3': np.array([0.20, 0.12, 0.10, 0.04, 0.03])}  # best arm is 0, no searched technology terms
bids_to_clicks = {'C1': np.array([100, 3]),
                  'C2': np.array([90, 1]),
                  'C3': np.array([70, 0.5])}
bids_to_cum_costs = {'C1': np.array([35, 1]),
                     'C2': np.array([25, 0.5]),
                     'C3': np.array([15, 0.2])}
other_costs = 400

# Bids setup
n_bids = 100
min_bid = 0.5
max_bid = 10.0
sigma = 2

# Parameters for learners
# SW-UCB
window_size = 4 * int(np.sqrt(T))
# CUSUM-UCB
M = 50
eps = 0.1
h = 0.5 * np.log(T)
alpha = np.sqrt(np.log(T) / T)

# Parameters for step 5 and 6_1
phases_duration = [120, 122, 123]
probabilities_step5 = {'C1': np.array([0.07, 0.10, 0.10, 0.20, 0.10]),  # best arm is 3 (starting from 0)
                       'C2': np.array([0.08, 0.06, 0.12, 0.03, 0.03]),  # best arm is 2
                       'C3': np.array([0.18, 0.30, 0.15, 0.02, 0.02])}  # best arm is 1
bids_to_clicks_cost = {'C1': np.array([100, 3]),  # this curve doesn't change
                       'C2': np.array([100, 3]),
                       'C3': np.array([100, 3])}
bids_to_cum_costs_cost = {'C1': np.array([35, 1]),  # this curve doesn't change
                          'C2': np.array([35, 1]),
                          'C3': np.array([35, 1])}

# Parameters for step 6_2
phases_duration_step6 = [16, 21, 23, 15, 19]
bid_idx = 25  # 2.9
prices_step6 = {'C1': np.array([500, 550, 600, 650, 700]),
                'C2': np.array([500, 550, 600, 650, 700]),
                'C3': np.array([500, 550, 600, 650, 700]),
                'C4': np.array([500, 550, 600, 650, 700]),
                'C5': np.array([500, 550, 600, 650, 700])}
probabilities_step6 = {'C1': np.array([0.07, 0.10, 0.10, 0.20, 0.10]),  # best arm is 3 (starting from 0)
                       'C2': np.array([0.08, 0.06, 0.12, 0.03, 0.03]),  # best arm is 2
                       'C3': np.array([0.18, 0.30, 0.15, 0.02, 0.02]),  # best arm is 1
                       'C4': np.array([0.27, 0.12, 0.06, 0.02, 0.01]),  # best arm is 0
                       'C5': np.array([0.09, 0.06, 0.12, 0.10, 0.13])}  # best arm is 4
bids_to_clicks_cost_step6 = {'C1': np.array([100, 2]),  # this curve doesn't change
                             'C2': np.array([100, 2]),
                             'C3': np.array([100, 2]),
                             'C4': np.array([100, 2]),
                             'C5': np.array([100, 2])}
bids_to_cum_costs_cost_step6 = {'C1': np.array([20, 0.5]),  # this curve doesn't change
                                'C2': np.array([20, 0.5]),
                                'C3': np.array([20, 0.5]),
                                'C4': np.array([20, 0.5]),
                                'C5': np.array([20, 0.5])}


def iterate_over_counter(counter, reference_array):
    for key, value in counter.items():
        print(f"{reference_array[key]}, index {key}, is the best in {value} experiments")

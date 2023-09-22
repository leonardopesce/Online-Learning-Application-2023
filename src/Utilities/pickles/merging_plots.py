import numpy as np

from src import settings

from src.Environments import Environment
from src.Learners import Clairvoyant
from src.Utilities import plot_all_algorithms, plot_all_algorithms_divided
import pickle

T = 365
algorithms = ['UCB', 'TS']
categories = ['C1', 'C2', 'C3']

env = Environment(settings.n_prices, settings.prices, settings.probabilities, settings.bids_to_clicks, settings.bids_to_cum_costs, settings.other_costs)
clairvoyant = Clairvoyant(env)
best_rewards = np.array([])

best_reward = 0
for category in categories:
    _, _, _, _, best_reward_category = clairvoyant.maximize_reward(category)
    best_reward += best_reward_category
best_rewards = np.append(best_rewards, np.ones((T,)) * best_reward)

fileObj1 = open('learners_step4_scenario1.pkl', 'rb')
fileObj2 = open('learners_step4_scenario2.pickle', 'rb')
fileObj3 = open('learners_step4_scenario3.pkl', 'rb')

read1 = pickle.load(fileObj1)
read2 = pickle.load(fileObj2)
read3 = pickle.load(fileObj3)

reward_per_algorithm1 = [np.sum(np.array([read1[algorithm][category] for category in categories]), axis=0) for algorithm in algorithms]
ucb1 = reward_per_algorithm1[0]
ts1 = reward_per_algorithm1[1]
ucb1 = [ucb1[i] for i in range(ucb1.shape[0])]
ts1 = [ts1[i] for i in range(ts1.shape[0])]
reward_per_algorithm1 = [ucb1, ts1]
#reward_per_algorithm2 = [read2[algorithm] for algorithm in algorithms]
reward_per_algorithm2 = read2
reward_per_algorithm3 = [read3[algorithm] for algorithm in algorithms]

fileObj1.close()
fileObj2.close()
fileObj3.close()

algorithms = ['UCB known context', 'TS known context', 'UCB', 'TS']

plot_all_algorithms(reward_per_algorithm1 + reward_per_algorithm2, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_merged12")
plot_all_algorithms_divided(reward_per_algorithm1 + reward_per_algorithm2, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_merged12")

algorithms = ['UCB', 'TS', 'UCB single learner', 'TS single learner']

plot_all_algorithms(reward_per_algorithm2 + reward_per_algorithm3, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_merged23")
plot_all_algorithms_divided(reward_per_algorithm2 + reward_per_algorithm3, best_rewards, np.arange(0, T, 1), algorithms, step_name="step4_merged23")

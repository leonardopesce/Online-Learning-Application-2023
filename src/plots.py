import numpy as np
import matplotlib.pyplot as plt


def plot_instantaneous_regret(axes, rewards, best_rewards, label, x_range):
    """
    Plot the instantaneous regret of a single algorithm inside a subplots graph

    :param matplotlib.Axes axes: Axes where to plot the graphs
    :param list rewards: Array of reward of the algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param str label: Name of the algorithm plotted
    :param np.ndarray x_range: Array with round values
    """

    regret_mean = np.mean(best_rewards - rewards, axis=0)
    regret_std = np.std(best_rewards - rewards, axis=0)
    axes[0].set_title(f'Instantaneous Regret {label}')
    axes[0].plot(regret_mean, 'r')
    axes[0].fill_between(x_range, regret_mean - regret_std, regret_mean + regret_std, color='r', alpha=0.2)
    axes[0].axhline(y=0, color='b', linestyle='--')
    axes[0].legend([f"{label} mean", f"{label}std"])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Instantaneous regret")


def plot_instantaneous_reward(axes, rewards, best_rewards, label, x_range):
    """
    Plot the instantaneous reward of a single algorithm inside a subplots graph

    :param matplotlib.Axes axes: Axes where to plot the graphs
    :param list rewards: Array of reward of the algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param str label: Name of the algorithm plotted
    :param np.ndarray x_range: Array with round values
    """

    reward_mean = np.mean(rewards, axis=0)
    reward_std = np.std(rewards, axis=0)
    axes[1].set_title(f'Instantaneous reward plot for {label}')
    axes[1].plot(reward_mean, 'r')
    axes[1].fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, color='r', alpha=0.2)
    axes[1].plot(best_rewards, 'b')
    axes[1].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Instantaneous reward")


def plot_cumulative_regret(axes, rewards, best_rewards, label, x_range):
    """
    Plot the cumulative regret of a single algorithm inside a subplots graph

    :param matplotlib.Axes axes: Axes where to plot the graphs
    :param list rewards: Array of reward of the algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param str label: Name of the algorithm plotted
    :param np.ndarray x_range: Array with round values
    """

    cum_regret_mean = np.mean(np.cumsum(best_rewards - rewards, axis=1), axis=0)
    cum_regret_std = np.std(np.cumsum(best_rewards - rewards, axis=1), axis=0)
    axes[2].set_title(f'Cumulative regret plot for {label}')
    axes[2].plot(cum_regret_mean, 'r')
    axes[2].fill_between(x_range, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, color='r', alpha=0.2)
    axes[2].legend([f"{label} mean", f"{label} std"])
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("Cumulative regret")


def plot_cumulative_reward(axes, rewards, best_rewards, label, x_range):
    """
    Plot the cumulative reward of a single algorithm inside a subplots graph

    :param matplotlib.Axes axes: Axes where to plot the graphs
    :param list rewards: Array of reward of the algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param str label: Name of the algorithm plotted
    :param np.ndarray x_range: Array with round values
    """

    cum_reward_mean = np.mean(np.cumsum(rewards, axis=1), axis=0)
    cum_reward_std = np.std(np.cumsum(rewards, axis=1), axis=0)
    axes[3].set_title(f'Cumulative reward plot for {label}')
    axes[3].plot(cum_reward_mean, 'r')
    axes[3].fill_between(x_range, cum_reward_mean - cum_reward_std, cum_reward_mean + cum_reward_std, color='r', alpha=0.2)
    axes[3].plot(np.cumsum(best_rewards), 'b')
    axes[3].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("Cumulative reward")


def plot_single_algorithm(reward_per_experiment, best_rewards, label, x_range):
    """
    Plot the graphs of different algorithms is different subplots

    :param list reward_per_experiment: Array of reward of the algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param str label: Name of the algorithm plotted
    :param np.ndarray x_range: Array with round values
    """

    # For the given algorithm it is plotted a plot with four different graphs
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    plot_instantaneous_regret(axes=axes, rewards=reward_per_experiment, best_rewards=best_rewards, label=label, x_range=x_range)
    plot_instantaneous_reward(axes=axes, rewards=reward_per_experiment, best_rewards=best_rewards, label=label, x_range=x_range)
    plot_cumulative_regret(axes=axes, rewards=reward_per_experiment, best_rewards=best_rewards, label=label, x_range=x_range)
    plot_cumulative_reward(axes=axes, rewards=reward_per_experiment, best_rewards=best_rewards, label=label, x_range=x_range)
    plt.show()


def plot_all_algorithms(reward_per_algorithm, best_rewards, x_range, labels, step_name=''):
    """
    Plot a graphs with all the algorithms

    :param list reward_per_algorithm: Array of reward of each algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param np.ndarray x_range: Array with round values
    :param list labels: List of names of algorithms plotted
    :param str step_name: Name of the step of the project
    """

    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    #colors = ['#35879b', '#ba4f35']
    colors = ["#D00000", "#3335FF", "#FFBA08", "#94C9A9", "#372554"]
    # Plot the instantaneous regret of all the algorithms
    axes[0].set_title('Instantaneous regret plot', fontsize=20)
    for i, label in enumerate(labels):
        regret_mean = np.mean(best_rewards - reward_per_algorithm[i], axis=0)
        regret_std = np.std(best_rewards - reward_per_algorithm[i], axis=0)
        axes[0].plot(regret_mean, label=label, color=colors[i])
        axes[0].fill_between(x_range, regret_mean - regret_std, regret_mean + regret_std, alpha=0.35, color=colors[i])
    axes[0].axhline(y=0, color='black', linestyle='--')
    axes[0].legend(fontsize=18)
    axes[0].set_xlabel("t", fontsize=18)
    axes[0].set_ylabel("Instantaneous regret", fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].yaxis.get_offset_text().set_fontsize(18)

    # Plot the instantaneous reward of all the algorithms
    axes[1].set_title('Instantaneous reward plot', fontsize=20)
    for i, label in enumerate(labels):
        reward_mean = np.mean(reward_per_algorithm[i], axis=0)
        reward_std = np.std(reward_per_algorithm[i], axis=0)
        axes[1].plot(reward_mean, label=label, color=colors[i])
        axes[1].fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, alpha=0.35, color=colors[i])
    axes[1].plot(best_rewards, 'black', label='Clairvoyant')
    axes[1].legend(fontsize=18)
    axes[1].set_xlabel("t", fontsize=18)
    axes[1].set_ylabel("Instantaneous reward", fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].yaxis.get_offset_text().set_fontsize(18)

    # Plot the cumulative regret of all the algorithms
    axes[2].set_title('Cumulative regret plot', fontsize=20)
    for i, label in enumerate(labels):
        cum_regret_mean = np.mean(np.cumsum(best_rewards - reward_per_algorithm[i], axis=1), axis=0)
        cum_regret_std = np.std(np.cumsum(best_rewards - reward_per_algorithm[i], axis=1), axis=0)
        axes[2].plot(cum_regret_mean, label=label, color=colors[i])
        axes[2].fill_between(x_range, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, alpha=0.35, color=colors[i])
    axes[2].legend(fontsize=18)
    axes[2].set_xlabel("t", fontsize=18)
    axes[2].set_ylabel("Cumulative regret", fontsize=18)
    axes[2].tick_params(axis='both', which='major', labelsize=18)
    axes[2].yaxis.get_offset_text().set_fontsize(18)

    # Plot the cumulative reward of all the algorithms
    axes[3].set_title('Cumulative reward plot', fontsize=20)
    for i, label in enumerate(labels):
        cum_reward_mean = np.mean(np.cumsum(reward_per_algorithm[i], axis=1), axis=0)
        cum_reward_std = np.std(np.cumsum(reward_per_algorithm[i], axis=1), axis=0)
        axes[3].plot(cum_reward_mean, label=label, color=colors[i])
        axes[3].fill_between(x_range, cum_reward_mean - cum_reward_std, cum_reward_mean + cum_reward_std, alpha=0.35, color=colors[i])
    axes[3].plot(np.cumsum(best_rewards), 'black', label='Clairvoyant')
    axes[3].legend(fontsize=18)
    axes[3].set_xlabel("t", fontsize=18)
    axes[3].set_ylabel("Cumulative reward", fontsize=18)
    axes[3].tick_params(axis='both', which='both', labelsize=18)
    axes[3].yaxis.get_offset_text().set_fontsize(18)

    plt.savefig('../img/plots_all_' + step_name + '.png', bbox_inches='tight')
    plt.show()


def plot_clicks_curve(bids, learners, labels, original, additional_label='', step_name=''):
    """
    Plot the estimate of the number of daily clicks curve

    :param np.ndarray bids: Array with the possible values of bids
    :param dict learners: Dictionary with arrays with the GP learners
    :param list labels: List of names of algorithms plotted
    :param ndarray original: y-values of the original curve
    :param str additional_label: Additional label to add to the label of the algorithms
    :param str step_name: Name of the step of the project
    """

    plt.figure(0)

    for label in labels:
        mean_clicks_per_experiment = np.mean(np.array([learner.means_clicks for learner in learners[label]]), axis=0)
        lower_bounds_clicks_per_experiment = np.mean(np.array([learner.lower_bounds_clicks for learner in learners[label]]), axis=0)
        upper_bounds_clicks_per_experiment = np.mean(np.array([learner.upper_bounds_clicks for learner in learners[label]]), axis=0)

        plt.plot(bids, mean_clicks_per_experiment, label='GP-'+label+additional_label)
        plt.fill_between(bids, lower_bounds_clicks_per_experiment, upper_bounds_clicks_per_experiment, alpha=0.2)

    plt.plot(bids, original, label='Original curve')
    plt.title('Clicks given the bid - GP' + additional_label, fontsize=18)
    plt.xlabel('Bids', fontsize=15)
    plt.ylabel('Number of clicks', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='both', labelsize=15)
    plt.savefig('../img/plot_clicks_' + step_name + '.png', bbox_inches='tight')
    plt.show()


def plot_costs_curve(bids, learners, labels, original, additional_label='', step_name=''):
    """
    Plot the estimate of the cumulative daily cost of the clicks curve

    :param np.ndarray bids: Array with the possible values of bids
    :param dict learners: Dictionary with arrays with the GP learners
    :param list labels: List of names of algorithms plotted
    :param ndarray original: y-values of the original curve
    :param str additional_label: Additional label to add to the label of the algorithms
    :param str step_name: Name of the step of the project
    """

    plt.figure(1)

    for label in labels:
        mean_cum_costs_per_experiment = np.mean(np.array([learner.means_costs for learner in learners[label]]), axis=0)
        lower_bounds_costs_per_experiment = np.mean(np.array([learner.lower_bounds_costs for learner in learners[label]]), axis=0)
        upper_bounds_costs_per_experiment = np.mean(np.array([learner.upper_bounds_costs for learner in learners[label]]), axis=0)

        plt.plot(bids, mean_cum_costs_per_experiment, label='GP-'+label+additional_label)
        plt.fill_between(bids, lower_bounds_costs_per_experiment, upper_bounds_costs_per_experiment, alpha=0.2)

    plt.plot(bids, original, label='Original curve')
    plt.title('Cost of the clicks given the bid - GP' + additional_label, fontsize=18)
    plt.xlabel('Bids', fontsize=15)
    plt.ylabel('Cumulative cost', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='both', labelsize=15)
    plt.savefig('../img/plot_costs' + step_name + '.png', bbox_inches='tight')
    plt.show()


def plot_all_algorithms_divided(reward_per_algorithm, best_rewards, x_range, labels, step_name=''):
    """
    Plot a graphs with all the algorithms

    :param list reward_per_algorithm: Array of reward of each algorithm of each round
    :param np.ndarray best_rewards: Array of best reward of each round
    :param np.ndarray x_range: Array with round values
    :param list labels: List of names of algorithms plotted
    :param str step_name: Name of the step of the project
    """


    #colors = ['#35879b', '#ba4f35']
    colors = ["#D00000", "#3335FF", "#FFBA08", "#94C9A9", "#372554"]
    # Plot the instantaneous regret of all the algorithms
    plt.title('Instantaneous regret plot', fontsize=20)
    for i, label in enumerate(labels):
        regret_mean = np.mean(best_rewards - reward_per_algorithm[i], axis=0)
        regret_std = np.std(best_rewards - reward_per_algorithm[i], axis=0)
        plt.plot(regret_mean, label=label, color=colors[i])
        plt.fill_between(x_range, regret_mean - regret_std, regret_mean + regret_std, alpha=0.35, color=colors[i])
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.ylabel("Instantaneous regret", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
    plt.savefig('../img/plot_instantaneous_regret_' + step_name + '.png', bbox_inches='tight')
    plt.show()

    # Plot the instantaneous reward of all the algorithms
    plt.title('Instantaneous reward plot', fontsize=20)
    for i, label in enumerate(labels):
        reward_mean = np.mean(reward_per_algorithm[i], axis=0)
        reward_std = np.std(reward_per_algorithm[i], axis=0)
        plt.plot(reward_mean, label=label, color=colors[i])
        plt.fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, alpha=0.35, color=colors[i])
    plt.plot(best_rewards, 'black', label='Clairvoyant')
    plt.legend(fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.ylabel("Instantaneous reward", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
    plt.savefig('../img/plot_instantaneous_reward_' + step_name + '.png', bbox_inches='tight')
    plt.show()

    # Plot the cumulative regret of all the algorithms
    plt.title('Cumulative regret plot', fontsize=20)
    for i, label in enumerate(labels):
        cum_regret_mean = np.mean(np.cumsum(best_rewards - reward_per_algorithm[i], axis=1), axis=0)
        cum_regret_std = np.std(np.cumsum(best_rewards - reward_per_algorithm[i], axis=1), axis=0)
        plt.plot(cum_regret_mean, label=label, color=colors[i])
        plt.fill_between(x_range, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, alpha=0.35, color=colors[i])
    plt.legend(fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.ylabel("Cumulative regret", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
    plt.savefig('../img/plot_cumulative_regret_' + step_name + '.png', bbox_inches='tight')
    plt.show()

    # Plot the cumulative reward of all the algorithms
    plt.title('Cumulative reward plot', fontsize=20)
    for i, label in enumerate(labels):
        cum_reward_mean = np.mean(np.cumsum(reward_per_algorithm[i], axis=1), axis=0)
        cum_reward_std = np.std(np.cumsum(reward_per_algorithm[i], axis=1), axis=0)
        plt.plot(cum_reward_mean, label=label, color=colors[i])
        plt.fill_between(x_range, cum_reward_mean - cum_reward_std, cum_reward_mean + cum_reward_std, alpha=0.35, color=colors[i])
    plt.plot(np.cumsum(best_rewards), 'black', label='Clairvoyant')
    plt.legend(fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.ylabel("Cumulative reward", fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=18)
    plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
    plt.savefig('../img/plot_cumulative_reward_' + step_name + '.png', bbox_inches='tight')
    plt.show()


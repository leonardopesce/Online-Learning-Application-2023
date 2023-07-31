import numpy as np
import matplotlib.pyplot as plt

class Plots():

    def plot_instantaneous_regret(self, axes, regret_mean, regret_std, label, x_range):
        """
            Plot the instantaneous regret of a single algorithm inside a subplots graph

            :param matplotlib.Axes axes: axes where to plot the graphs
            :param np.ndarray regret_mean: array of regrets' mean of each round
            :param np.ndarray regret_std: array of regrets' std of each round
            :param str label: name of the algorithm plotted
            :param np.arange x_range: array with round values
        """
        axes[0].set_title(f'Instantaneous Regret {label}')
        axes[0].plot(regret_mean, 'r')
        axes[0].fill_between(x_range, regret_mean - regret_std, regret_mean + regret_std, color='r', alpha=0.2)
        axes[0].axhline(y=0, color='b', linestyle='--')
        axes[0].legend([f"{label} mean", f"{label}std"])
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Instantaneous regret")

    def plot_instantaneous_reward(self, axes, reward_mean, reward_std, best_reward, label, x_range):
        """
            Plot the instantaneous reward of a single algorithm inside a subplots graph

            :param: matplotlib.Axes axes: axes to plot the graphs
            :param: np.ndarray reward_mean: array of rewards' mean of each round
            :param: np.ndarray reward_std: array of rewards' std of each round
            :param: np.ndarray best_reward: array of best reward of each round
            :param: str label: name of the algorithm plotted
            :param: np.arange x_range: array with round values
        """
        axes[1].set_title(f'Instantaneous reward plot for {label}')
        axes[1].plot(reward_mean, 'r')
        axes[1].fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, color='r', alpha=0.2)
        axes[1].plot(best_reward, 'b')
        axes[1].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Instantaneous reward")

    def plot_cumulative_regret(self, axes, cum_regret_mean, cum_regret_std, label, x_range):
        """
            Plot the cumulative regret of a single algorithm inside a subplots graph

            :param: matplotlib.Axes axes: axes to plot the graphs
            :param: np.ndarray cum_regret_mean: array of cumulative regrets' mean of each round
            :param: np.ndarray cum_regret_std: array of cumulative regrets' std of each round
            :param: str label: name of the algorithm plotted
            :param: np.arange x_range: array with round values
        """
        axes[2].set_title(f'Cumulative regret plot for {label}')
        axes[2].plot(cum_regret_mean, 'r')
        axes[2].fill_between(x_range, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, color='r', alpha=0.2)
        axes[2].legend([f"{label} mean", f"{label} std"])
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("Cumulative regret")

    def plot_cumulative_reward(self, axes, cum_reward_mean, cum_reward_std, best_reward, label, x_range):
        """
            Plot the cumulative reward of a single algorithm inside a subplots graph

            :param: matplotlib.Axes axes: axes to plot the graphs
            :param: np.ndarray cum_reward_mean: array of cumulative rewards' mean of each round
            :param: np.ndarray cum_reward_std: array of cumulative rewards' std of each round
            :param: np.ndarray best_reward: array of best reward of each round
            :param: str label: name of the algorithm plotted
            :param: np.arange x_range: array with round values
        """
        axes[3].set_title(f'Cumulative reward plot for {label}')
        axes[3].plot(cum_reward_mean, 'r')
        axes[3].fill_between(x_range, cum_reward_mean - cum_reward_std, cum_reward_mean + cum_reward_std, color='r', alpha=0.2)
        axes[3].plot(np.cumsum(best_reward), 'b')
        axes[3].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
        axes[3].set_xlabel("t")
        axes[3].set_ylabel("Cumulative reward")

    def plot_single_algorithms(self, regret_means, regret_stds, cum_regret_means, cum_regret_stds, reward_means, reward_stds, cum_reward_means, cum_reward_stds, best_reward, legend, x_range):
        """
            Plot the graphs of different algorithms is different subplots

            :param: np.ndarray regret_means: array of the regret means of each algorithm
            :param: np.ndarray regret_stds: array of the regret std of each algorithm
            :param: np.ndarray cum_regret_means: array of the cumulative regret mean of each algorithm
            :param: np.ndarray cum_regret_stds: array of the cumulative regret std of each algorithm
            :param: np.ndarray reward_means: array of the reward mean of each algorithm
            :param: np.ndarray reward_stds: array of the reward std of each algorithm
            :param: np.ndarray cum_reward_means: array of the cumulative reward of each algorithm
            :param: np.ndarray cum_reward_stds: array of the cumulative reward std of each algorithm
            :param: np.ndarray best_reward: array of best reward of each round
            :param: list legend: list of names of algorithms plotted
            :param: np.arange x_range: array with round values
        """

        # Check that each array has the same length as the length of the legend
        n_plots = len(legend)
        assert len(regret_means) == n_plots
        assert len(regret_stds) == n_plots
        assert len(cum_regret_means) == n_plots
        assert len(cum_regret_stds) == n_plots
        assert len(reward_means) == n_plots
        assert len(regret_stds) == n_plots
        assert len(cum_reward_means) == n_plots
        assert len(cum_reward_stds) == n_plots

        # For each algorithm it is plotted a plot with four different graphs
        for i in range(n_plots):
            _, axes = plt.subplots(2, 2, figsize=(20, 20))
            axes = axes.flatten()

            self.plot_instantaneous_regret(axes=axes, regret_mean=regret_means[i], regret_std=regret_stds[i], label=legend[i], x_range=x_range)
            self.plot_instantaneous_reward(axes=axes, reward_mean=reward_means[i], reward_std=reward_stds[i], best_reward=best_reward, label=legend[i], x_range=x_range)
            self.plot_cumulative_regret(axes=axes, cum_regret_mean=cum_regret_means[i], cum_regret_std=cum_regret_stds[i], label=legend[i], x_range=x_range)
            self.plot_cumulative_reward(axes=axes, cum_reward_mean=cum_reward_means[i], cum_reward_std=cum_reward_stds[i], best_reward=best_reward, label=legend[i], x_range=x_range)
            plt.show()

    def plot_all_algorithms(self, regret_means, cum_regret_means, reward_means, cum_reward_means, best_reward, legend):
        """
            Plot a graphs with all the algorithms

            :param: np.ndarray regret_means: array of the regret means of each algorithm
            :param: np.ndarray cum_regret_means: array of the cumulative regret mean of each algorithm
            :param: np.ndarray reward_means: array of the reward mean of each algorithm
            :param: np.ndarray cum_reward_means: array of the cumulative reward of each algorithm
            :param: np.ndarray best_reward: array of best reward of each round
            :param: list legend: list of names of algorithms plotted
        """
        n_plots = len(legend)
        _, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()

        # Plot the instantaneous regret of all the algorithms
        axes[0].set_title('Instantaneous regret plot')
        for i in range(n_plots):
            axes[0].plot(regret_means[i])
        axes[0].axhline(y=0, color='b', linestyle='--')
        axes[0].legend(legend)
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Instantaneous regret")

        # Plot the Instantaneous reward of all the algorithms
        axes[1].set_title('Instantaneous reward plot')
        for i in range(n_plots):
            axes[1].plot(reward_means[i])
        axes[1].plot(best_reward, 'b')
        legend.append('Clairvoyant')
        axes[1].legend(legend)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Instantaneous reward")

        # Plot the cumulative regret of all the algorithms
        axes[2].set_title('Cumulative regret plot')
        for i in range(n_plots):
            axes[2].plot(cum_regret_means[i])
        axes[2].legend(legend)
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("Cumulative regret")

        # Plot the cumulative reward of all the algorithms
        axes[3].set_title('Cumulative reward plot')
        for i in range(n_plots):
            axes[3].plot(cum_reward_means[i])
        axes[3].plot(np.cumsum(best_reward), 'b')
        legend.append('Clairvoyant')
        axes[3].legend(legend)
        axes[3].set_xlabel("t")
        axes[3].set_ylabel("Cumulative reward")
        plt.show()

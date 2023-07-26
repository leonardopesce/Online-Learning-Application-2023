import numpy as np
import matplotlib.pyplot as plt

class Plots():

    def plot_instantaneous_regret(self, axes, regret_mean, regret_std, label, x_range):
        axes[0].set_title(f'Instantaneous Regret {label}')
        axes[0].plot(regret_mean, 'b')
        axes[0].fill_between(x_range, regret_mean - regret_std, regret_mean + regret_std, color='b', alpha=0.2)
        axes[0].axhline(y=0, color='b', linestyle='--')
        axes[0].legend([f"{label} mean", f"{label}std"])
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Instantaneous regret")

    def plot_instantaneous_reward(self, axes, reward_mean, reward_std, best_reward, label, x_range):
        axes[1].set_title(f'Instantaneous reward plot for {label}')
        axes[1].plot(reward_mean, 'r')
        axes[1].fill_between(x_range, reward_mean - reward_std, reward_mean + reward_std, color='r', alpha=0.2)
        axes[1].plot(best_reward, 'b')
        axes[1].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Instantaneous reward")

    def plot_cumulative_regret(self, axes, cum_regret_mean, cum_regret_std, label, x_range):
        axes[2].set_title(f'Cumulative regret plot for {label}')
        axes[2].plot(cum_regret_mean, 'b')
        axes[2].fill_between(x_range, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, color='b', alpha=0.2)
        axes[2].legend([f"{label} mean", f"{label} std"])
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("Cumulative regret")

    def plot_cumulative_reward(self, axes, cum_reward_mean, cum_reward_std, best_reward, label, x_range):
        axes[3].set_title(f'Cumulative reward plot for {label}')
        axes[3].plot(cum_reward_mean, 'r')
        axes[3].fill_between(x_range, cum_reward_mean - cum_reward_std, cum_reward_mean + cum_reward_std, color='r', alpha=0.2)
        axes[3].plot(np.cumsum(best_reward), 'b')
        axes[3].legend([f"{label} mean", f"{label} std", "Clairvoyant"])
        axes[3].set_xlabel("t")
        axes[3].set_ylabel("Cumulative reward")

    def plot_single_algorithms(self, regret_means, regret_stds, cum_regret_means, cum_regret_stds, reward_means, reward_stds, cum_reward_means, cum_reward_stds, best_reward, legend, x_range):
        n_plots = len(legend)
        assert len(regret_means) == n_plots
        assert len(regret_stds) == n_plots
        assert len(cum_regret_means) == n_plots
        assert len(cum_regret_stds) == n_plots
        assert len(reward_means) == n_plots
        assert len(regret_stds) == n_plots
        assert len(cum_reward_means) == n_plots
        assert len(cum_reward_stds) == n_plots

        for i in range(n_plots):
            _, axes = plt.subplots(2, 2, figsize=(20, 20))
            axes = axes.flatten()

            self.plot_instantaneous_regret(axes=axes, regret_mean=regret_means[i], regret_std=regret_stds[i], label=legend[i], x_range=x_range)
            self.plot_instantaneous_reward(axes=axes, reward_mean=reward_means[i], reward_std=reward_stds[i], best_reward=best_reward, label=legend[i], x_range=x_range)
            self.plot_cumulative_regret(axes=axes, cum_regret_mean=cum_regret_means[i], cum_regret_std=cum_regret_stds[i], label=legend[i], x_range=x_range)
            self.plot_cumulative_reward(axes=axes, cum_reward_mean=cum_reward_means[i], cum_reward_std=cum_reward_stds[i], best_reward=best_reward, label=legend[i], x_range=x_range)
            plt.show()

    def plot_all_algorithms(self, regret_means, cum_regret_means, reward_means, cum_reward_means, best_reward, legend):
        n_plots = len(legend)
        _, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()

        axes[0].set_title('Instantaneous regret plot')
        for i in range(n_plots):
            axes[0].plot(regret_means[i])
        axes[0].axhline(y=0, color='b', linestyle='--')
        axes[0].legend(legend)
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Instantaneous regret")

        axes[1].set_title('Instantaneous reward plot')
        for i in range(n_plots):
            axes[1].plot(reward_means[i])
        axes[1].plot(best_reward, 'b')
        legend.append('Clairvoyant')
        axes[1].legend(legend)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Instantaneous reward")

        axes[2].set_title('Cumulative regret plot')
        for i in range(n_plots):
            axes[2].plot(cum_regret_means[i])
        axes[2].legend(legend)
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("Cumulative regret")

        axes[3].set_title('Cumulative reward plot')
        for i in range(n_plots):
            axes[3].plot(cum_reward_means[i])
        axes[3].plot(np.cumsum(best_reward), 'b')
        legend.append('Clairvoyant')
        axes[3].legend(legend)
        axes[3].set_xlabel("t")
        axes[3].set_ylabel("Cumulative reward")
        plt.show()
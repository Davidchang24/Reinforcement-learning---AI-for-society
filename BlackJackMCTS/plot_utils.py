import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_blackjack_values(V):
    def get_Z(x, y, usable_ace):
        return V.get((x, y, usable_ace), 0)

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.set_zlabel("State Value")
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.set_title('Usable Ace')
    get_figure(True, ax1)
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.set_title('No Usable Ace')
    get_figure(False, ax2)
    plt.show()

def plot_policy(policy):
    def get_Z(x, y, usable_ace):
        return policy.get((x, y, usable_ace), 1)

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for y in y_range] for x in x_range])

        ax.imshow(Z.T, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[11, 21, 1, 10], origin='lower')
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(ax.images[0], ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(121)
    ax1.set_title('Usable Ace')
    get_figure(True, ax1)
    ax2 = fig.add_subplot(122)
    ax2.set_title('No Usable Ace')
    get_figure(False, ax2)
    plt.show()

def plot_win_rate(rewards_all_episodes, num_episodes):
    rewards_optimal = np.array(rewards_all_episodes)
    rewards_optimal[rewards_optimal == -1] = 0
    rewards_optimal = rewards_optimal.cumsum()
    win_rate_optimal = rewards_optimal / np.arange(1, num_episodes + 1)

    plt.plot(np.arange(1, num_episodes + 1), win_rate_optimal, label='Optimal Policy')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rate over Time')
    plt.show()

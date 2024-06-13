import numpy as np
import gymnasium as gym
from collections import defaultdict
import sys
import pickle
import os
from plot_utils import plot_policy, plot_win_rate

env = gym.make('Blackjack-v1')

NUM_EPISODES = 2000000
CHECKPOINT_INTERVAL = 1000000
C = 1
ALPHA = 0.03
CHECKPOINT_DIR = 'checkpoints'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

def default_q():
    return np.zeros(2)

def default_n():
    return np.zeros(2)

def monte_carlo_control(num_episodes, c, alpha):
    Q = defaultdict(default_q)
    N = defaultdict(default_n)
    rewards_all_episodes = np.zeros(num_episodes)
    policy = {}
    start_episode = 1

    # Load from checkpoint if it exists
    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint.pkl')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            Q = checkpoint['Q']
            N = checkpoint['N']
            rewards_all_episodes[:checkpoint['episode']] = checkpoint['rewards_all_episodes'][:checkpoint['episode']]
            start_episode = checkpoint['episode'] + 1
            policy = checkpoint['policy']
        print(f"Resuming from episode {start_episode}")

    for episode in range(start_episode, num_episodes + 1):
        if episode % 1000 == 0:
            print(f"\rEpisode {episode}/{num_episodes}.", end="")
            sys.stdout.flush()

        experience = generate_episode(Q, N, c, episode)
        states, actions, rewards = zip(*experience)
        rewards = np.array(rewards)
        rewards_all_episodes[episode - 1] = np.sum(rewards)
        G = 0

        visited = set()
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            if (state, action) not in visited:
                G = reward + G
                Q[state][action] += alpha * (G - Q[state][action])
                N[state][action] += 1
                visited.add((state, action))

        for state in states:
            policy[state] = np.argmax(Q[state])

        # Save checkpoint
        if episode % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_file, 'wb') as f:
                checkpoint = {
                    'Q': Q,
                    'N': N,
                    'rewards_all_episodes': rewards_all_episodes,
                    'episode': episode,
                    'policy': policy
                }
                pickle.dump(checkpoint, f)
            print(f"\nCheckpoint saved at episode {episode}")

    return Q, policy, rewards_all_episodes

def generate_episode(Q, N, c, episode):
    state = env.reset()[0]
    episode_data = []

    while True:
        action = ucb_policy(Q, N, state, c, episode)
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state

        if done:
            break

    return episode_data

def ucb_policy(Q, N, state, c, episode):
    total_count = np.sum(N[state])
    if total_count == 0:
        total_count = 1

    ucb_values = Q[state] + c * np.sqrt(np.log(episode + 1) / (N[state] + 1))
    return np.argmax(ucb_values)

if __name__ == "__main__":
    Q, policy, rewards_all_episodes = monte_carlo_control(NUM_EPISODES, C, ALPHA)
    print(f"\nEpisode {NUM_EPISODES}/{NUM_EPISODES}.")

    plot_win_rate(rewards_all_episodes, NUM_EPISODES)
    plot_policy(policy)

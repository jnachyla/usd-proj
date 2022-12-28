import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import collections
import statistics
import tqdm
import os

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


import ActorCritic
#from ActorCritic import train_step
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # Creating an environment
    env = gym.make("Pendulum-v0")

    # Set seed
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)


    num_actions = 1 # env.action_space.n
    num_hidden_units = 128

    model = ActorCritic.ActorCritic_class(num_actions, num_hidden_units)

    #% % time

    min_episodes_criterion = 100
    max_episodes = 1000
    max_steps_per_episode = 200

    # `CartPole-v1` is considered solved if average reward is >= 475 over 500
    # consecutive trials
    reward_threshold = -5
    running_reward = -16

    # The discount factor for future rewards
    gamma = 0.99

    # Keep the last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    t = tqdm.trange(max_episodes)
    for i in t:
        initial_state = env.reset() #usunelam info z lewej strony
        print(f'\nsprawdzenie - env.reset = {initial_state}')
        initial_state = tf.constant(initial_state, dtype=tf.float32)
        episode_reward = int(ActorCritic.ActorCritic_class.train_step(
            initial_state, model, ActorCritic.ActorCritic_class.optimizer, gamma, max_steps_per_episode))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
            pass  # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

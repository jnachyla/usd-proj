import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import collections
import statistics
import tqdm
import os

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Creating an environment
env = gym.make("Pendulum-v0")
env.reset()
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic_class(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

  def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, truncated, info = env.step(action)
    return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

  def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(ActorCritic_class.env_step, [action],
                           [tf.float32, tf.int32, tf.int32])

  def run_episode(
          initial_state: tf.Tensor,
          model: tf.keras.Model,
          max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
      """Runs a single episode to collect training data."""

      action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

      initial_state_shape = initial_state.shape
      state = initial_state

      for t in tf.range(max_steps):
          # Convert state into a batched tensor (batch size = 1)
          state = tf.expand_dims(state, 0)

          # Run the model and to get action probabilities and critic value
          action_logits_t, value = model(state)

          # Sample next action from the action probability distribution
          action = tf.random.categorical(action_logits_t, 1)[0, 0]
          action_probs_t = tf.nn.softmax(action_logits_t)

          # Store critic values
          values = values.write(t, tf.squeeze(value))

          # Store log probability of the action chosen
          action_probs = action_probs.write(t, action_probs_t[0, action])

          # Apply action to the environment to get next state and reward
          state, reward, done = ActorCritic_class.tf_env_step(action)
          state.set_shape(initial_state_shape)

          # Store reward
          rewards = rewards.write(t, reward)

          if tf.cast(done, tf.bool):
              break

      action_probs = action_probs.stack()
      values = values.stack()
      rewards = rewards.stack()

      return action_probs, values, rewards

  def get_expected_return(
          rewards: tf.Tensor,
          gamma: float,
          standardize: bool = True) -> tf.Tensor:
      """Compute expected returns per timestep."""

      n = tf.shape(rewards)[0]
      returns = tf.TensorArray(dtype=tf.float32, size=n)

      # Start from the end of `rewards` and accumulate reward sums
      # into the `returns` array
      rewards = tf.cast(rewards[::-1], dtype=tf.float32)
      discounted_sum = tf.constant(0.0)
      discounted_sum_shape = discounted_sum.shape
      for i in tf.range(n):
          reward = rewards[i]
          discounted_sum = reward + gamma * discounted_sum
          discounted_sum.set_shape(discounted_sum_shape)
          returns = returns.write(i, discounted_sum)
      returns = returns.stack()[::-1]

      if standardize:
          returns = ((returns - tf.math.reduce_mean(returns)) /
                     (tf.math.reduce_std(returns) + eps))

      return returns

  huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

  def compute_loss(
          action_probs: tf.Tensor,
          values: tf.Tensor,
          returns: tf.Tensor) -> tf.Tensor:
      """Computes the combined Actor-Critic loss."""

      advantage = returns - values

      action_log_probs = tf.math.log(action_probs)
      actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

      critic_loss = ActorCritic_class.huber_loss(values, returns)

      return actor_loss + critic_loss

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

  @tf.function
  def train_step(
          initial_state: tf.Tensor,
          model: tf.keras.Model,
          optimizer: tf.keras.optimizers.Optimizer,
          gamma: float,
          max_steps_per_episode: int) -> tf.Tensor:
      """Runs a model training step."""

      with tf.GradientTape() as tape:
          # Run the model for one episode to collect training data
          action_probs, values, rewards = ActorCritic_class.run_episode(
              initial_state, model, max_steps_per_episode)

          # Calculate the expected returns
          returns = ActorCritic_class.get_expected_return(rewards, gamma)

          # Convert training data to appropriate TF tensor shapes
          action_probs, values, returns = [
              tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

          # Calculate the loss values to update our network
          loss = ActorCritic_class.compute_loss(action_probs, values, returns)

      # Compute the gradients from the loss
      grads = tape.gradient(loss, model.trainable_variables)

      # Apply the gradients to the model's parameters
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      episode_reward = tf.math.reduce_sum(rewards)

      return episode_reward








import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import evaluators

env = gym.make("HalfCheetah-v4")
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("td3_halfcheetah")
#env = model.get_env()

#del model # remove to demonstrate saving and loading

model = TD3.load("td3_halfcheetah")

obs = env.reset()

evaluators.evaluate_policy(model, env, eval_episodes=100, render=False)




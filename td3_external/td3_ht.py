import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import evaluators

env = gym.make("HalfCheetah-v4")
# The noise objects for TD3
n_actions = env.action_space.shape[-1]


vals_policy = ["MLPPolicy", "CnnPolicy"]
vals_learning_rate = [2.0633e-05, 1.0e-03, 1.0e-07,1.0e03 ]
vals_buffer_size = [1000000, 10000, 10000, 100000000]
vals_train_freq = [1,4,1,10]
vals_tau=[0.01, 0.05, 0.0001, 1.0]
vals_policy_delay = [2,4,1,10]
vals_std_target_policy_noise = [0.1, 0.5, 0.001, 2.0 ]

def random_hiperparameter(name, v,  is_integer = False):
    mean = v[0]
    std = v[1]
    clip_from = v[2]
    clip_to = v[3]


    float_val = np.clip(np.random.normal(mean, std), clip_from, clip_to)

    print("Random {}  {}".format(name,float_val))
    if is_integer:
        return int(float_val)
    else:
        return float(float_val)

for i in range(5):

    lr = random_hiperparameter("learing_rate", vals_learning_rate, is_integer=False)
    buffer_size = random_hiperparameter("buffer_size", vals_buffer_size, is_integer=True)
    train_freq = random_hiperparameter("train_freq", vals_train_freq, is_integer=True)
    tau = random_hiperparameter("tau", vals_tau, is_integer=False)
    policy_delay = random_hiperparameter("policy_delay", vals_policy_delay, is_integer=True)
    std_target_policy_noise = random_hiperparameter("std_policy noise", vals_std_target_policy_noise, is_integer=False)

    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=std_target_policy_noise * np.ones(n_actions))

    env.reset()
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tau = tau,
                buffer_size=buffer_size, train_freq=train_freq,learning_rate=lr, policy_delay=policy_delay )
    model.learn(total_timesteps=500000, log_interval=10)
    model.save("td3_halfcheetah")
    env = model.get_env()
    env.reset()

    #del model # remove to demonstrate saving and loading


    model = TD3.load("td3_halfcheetah")

    obs = env.reset()

    res = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True)

    print("Mean")
    print(np.average(res[0]))
    print("STDev")
    print(np.std(res[0]))





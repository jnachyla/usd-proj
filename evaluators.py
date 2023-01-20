from tqdm import tqdm
import numpy as np

def evaluate_policy(model, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """

    avg_reward = 0.
    episode_max_length = 100000
    rewards = []
    for i in tqdm(range(eval_episodes)):
        obs = env.reset()
        done = False
        step = 0
        while not done or step > episode_max_length:
            if render:
                env.render()
            action = model.predict(obs)
            if  isinstance(action, tuple):

                obs, reward, done, _ = env.step(action[0])
            else:
                obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1
            rewards.append(reward)

    avg_reward /= eval_episodes
    std_dev = np.std(rewards)

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f} with standard deviation: {:f}".format(eval_episodes, avg_reward, std_dev))
    print("---------------------------------------")
    return avg_reward

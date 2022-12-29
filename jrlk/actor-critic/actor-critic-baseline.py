import gym
import tensorflow as tf
import tqdm as tqdm
from tensorflow import keras
from keras import layers
from keras import backend as K
import numpy as np

def get_actor(num_states, upper_bound):
    '''
    Zwraca sieć Aktora.
    Aproksymator funkcji polityki.

    Wejście: stany
    Wyjścia: akcje
    '''
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound

    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states):
    '''
    Zwraca sieć Krytyka.
    Aproksymator funkcji wartości dla polityki.

    Wejście: stany
    Wyjście: wartość wypłaty
    '''
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


class ActorCriticModel(object):
    '''
    Klasa modelu Aktor Krytyk - baseline.
    '''

    def __init__(self, env,  num_states, upper_bound, lower_bound):
        self.env = env

        self.gamma = 0.99

        self.optimizer_cri = keras.optimizers.Adam(learning_rate=0.01)
        self.optimizer_actor = keras.optimizers.Adam(learning_rate=0.01)

        # mniej wrażliwa na outliery funcja straty dla regresji
        # https: // en.wikipedia.org / wiki / Huber_loss
        self.huber_loss = keras.losses.Huber()

        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []

        self.running_reward = 0
        self.episode_count = 0
        self.num_episodes = 1000
        self.num_actions = env.action_space.shape[0]

        self.num_states = num_states
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        #tworzenie sieci
        self.critic_net = get_critic(num_states)
        self.actor_net = get_actor(num_states, upper_bound)

        self.critic_net.compile(optimizer = self.optimizer_cri)
        self.actor_net.compile(optimizer = self.optimizer_actor)
    def fun_target(self, reward, s_next):
        return reward + self.gamma * self.critic_net(s_next)

    def learn(self):

        running_reward = 0
        episode_count = 0
        while(True):
            state = self.env.reset()[0]
            episode_reward = 0
            actions = []

            # Update running reward to check condition for solving


            for timestep in tqdm.tqdm(range(1, self.num_episodes)):
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    # zamiana na tensor stanu
                    state = self.to_tensor(state)
                    # pobieramy akcje
                    action = self.choose_action(state)
                    # t
                    # wykonujemy akcje w chwili t
                    observation, reward, terminated, truncated, info = self.env.step(self.to_n_array(action))
                    # t+1
                    #liczymy poprawkę czasową

                    next_state = self.to_tensor(observation)

                    # poprawka czasowa
                    # target   =
                    # delta_value  = r(t + 1) + gamma * V(t + 1) - V(t)

                    target  = self.fun_target(reward, next_state)
                    critic_out = self.critic_net(state)
                    delta_value  = target - critic_out
                    delta_value = tf.squeeze(delta_value)
                    target = tf.squeeze(target)


                    # poprawka aktora
                    # Sample action from action probability distribution

                    # true_gradient = grad[logPi(s,a) * delta_value]
                    log_action = K.log(action)
                    loss_actor = K.sum(-log_action * delta_value)
                    #tape_actor.watch(self.actor_net.trainable_variables)

                    grads = tape_actor.gradient(loss_actor, self.actor_net.trainable_variables)
                    self.optimizer_actor.apply_gradients(zip(grads, self.actor_net.trainable_variables))


                    # poprawka krytyka
                    #tape_critic.watch(self.critic_net.trainable_variables)
                    loss_critic = self.huber_loss(target,critic_out)
                    grads_cri = tape_critic.gradient(loss_critic, self.critic_net.trainable_variables)
                    self.optimizer_cri.apply_gradients(zip(grads_cri, self.critic_net.trainable_variables))


                    episode_reward += reward
                    if timestep % 100 == 0:
                        template = "episode reward: {:.2f} at step {}"
                        print(template.format(episode_reward, timestep))

                    state = next_state

                    if terminated: break
            print("Episode reward = {}".format(episode_reward))
            # Log details
            episode_count += 1
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))

            if running_reward > 195:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

    def to_tensor(self, narray):

        tensor = tf.convert_to_tensor(narray)
        tensor = tf.squeeze(tensor)
        tensor = tf.expand_dims(tensor, 0)
        return tensor

    def to_n_array(self, tensor):
        return np.array([tensor.numpy()], dtype=float)


    def choose_action(self, state):
        squeeze = tf.squeeze(state)
        squeeze = tf.expand_dims(squeeze, 0)
        actor_out = tf.squeeze(self.actor_net(squeeze))

        # action = tf.random.truncated_normal(
        #     shape=[1],
        #     mean=actor_out,
        #     stddev=1.0,
        #     dtype=float
        # )

        # We make sure action is within bounds
        print(actor_out)
        legal_action = tf.clip_by_value(actor_out, self.lower_bound, self.upper_bound)

        return tf.squeeze(legal_action)


def run():
    # see https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    problem = "Pendulum-v1"
    problem2  = 'InvertedPendulum-v4'
    env = gym.make(problem)
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))


    ac = ActorCriticModel(env = env, num_states=num_states, upper_bound=upper_bound, lower_bound=lower_bound)
    ac.learn()

run()
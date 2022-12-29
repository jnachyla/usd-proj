import gym
import tensorflow as tf
import tqdm as tqdm
from tensorflow import keras
from keras import layers
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

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
    out = layers.Dense(30, activation="relu")(inputs)
    out = layers.Dense(20, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound

    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
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
    out = layers.Dense(30, activation="relu")(inputs)
    out = layers.Dense(20, activation="relu")(out)
    outputs = layers.Dense(1, kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
    return model

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula wziata z  https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class ActorCriticModel(object):
    '''
    Klasa modelu Aktor Krytyk - baseline.
    '''

    def __init__(self, env,  num_states, upper_bound, lower_bound, max_episodes, episode_len):
        self.episode_len = episode_len
        self.max_episodes = max_episodes
        self.env = env

        self.gamma = 0.9

        self.optimizer_cri = keras.optimizers.Adam(learning_rate=0.01)
        self.optimizer_actor = keras.optimizers.Adam(learning_rate=0.001)

        # mniej wrażliwa na outliery funcja straty dla regresji
        # https: // en.wikipedia.org / wiki / Huber_loss
        self.huber_loss = keras.losses.Huber()

        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.critic_losses_history = []
        self.actor_losses_history = []

        self.running_reward = 0
        self.episode_count = 0




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

        episode_count = 0

        for i in range(self.max_episodes):
            state = self.env.reset()[0]
            episode_reward = 0

            episode_losses_cri = []
            episode_losses_actor = []
            episode_rewards = []

            # Update running reward to check condition for solving

            for timestep in tqdm.tqdm(range(1, self.episode_len)):
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
                    # target   ~= r(t + 1) + gamma * V(t + 1)
                    # delta_value  = r(t + 1) + gamma * V(t + 1) - V(t)

                    target  = self.fun_target(reward, next_state)
                    critic_out = self.critic_net(state)
                    delta_value  = target - critic_out
                    delta_value = tf.squeeze(delta_value)
                    target = tf.squeeze(target)


                    # poprawka aktora
                    # true_gradient = grad[logPi(s,a) * delta_value]
                    log_action = K.log(action)
                    loss_actor = -log_action * delta_value


                    #tape_actor.watch(self.actor_net.trainable_variables)

                    grads = tape_actor.gradient(loss_actor, self.actor_net.trainable_variables)
                    self.optimizer_actor.apply_gradients(zip(grads, self.actor_net.trainable_variables))


                    # poprawka krytyka
                    #tape_critic.watch(self.critic_net.trainable_variables)
                    #loss_critic = self.huber_loss(target,critic_out)

                    grads_cri = tape_critic.gradient(delta_value, self.critic_net.trainable_variables)
                    self.optimizer_cri.apply_gradients(zip(grads_cri, self.critic_net.trainable_variables))

                    episode_rewards.append(reward)
                    if loss_actor == np.NAN:
                        print(loss_actor)
                    episode_losses_actor.append(loss_actor.numpy())
                    episode_losses_cri.append(delta_value.numpy())
                    episode_reward += reward
                    if timestep % 100 == 0:
                        template = "episode reward: {:.2f} at step {}"
                        print(template.format(episode_reward, timestep))

                    state = next_state

                    if terminated:
                        break
            print("Episode reward = {}".format(episode_reward))
            # Log details
            episode_count += 1

            self.rewards_history.append(episode_rewards)
            self.critic_losses_history.append(episode_losses_cri)
            self.actor_losses_history.append(episode_losses_actor)


        return self.rewards_history, self.actor_losses_history, self.critic_losses_history

    def to_tensor(self, narray):

        tensor = tf.convert_to_tensor(narray)
        tensor = tf.squeeze(tensor)
        tensor = tf.expand_dims(tensor, 0)
        return tensor

    def to_n_array(self, tensor):
        return np.array([tensor.numpy()], dtype=float)


    def choose_action(self, state, add_noise = False):
        squeeze = tf.squeeze(state)
        squeeze = tf.expand_dims(squeeze, 0)

        actor_out = tf.squeeze(self.actor_net(squeeze))
        if add_noise:
            ou_noise = OUActionNoise(mean=actor_out.numpy(), std_deviation=0.5)

            # We make sure action is within bounds
            action = actor_out + ou_noise()
        else:
            action = actor_out
        #print(action)
        legal_action = tf.clip_by_value(action, self.lower_bound, self.upper_bound)

        return tf.squeeze(legal_action)



def flatten(rewards_history):
    return [item for sublist in rewards_history for item in sublist]

def generate_x_vals(vals):
    return [i for i in range(len(vals))]

def draw_results(actor_losses_history, critic_losses_history, rewards_history):
    average_rewards = [sum(r) for r in rewards_history]
    all_rewards = flatten(rewards_history)
    average_losses_critic = [sum(r) / len(r) for r in critic_losses_history]
    all_losses_critic = flatten(critic_losses_history)
    average_losses_actor = [sum(r) / len(r) for r in actor_losses_history]
    all_losses_actor = flatten(actor_losses_history)
    fig, axs = plt.subplots(4)

    fig.suptitle('Actor Critic Baseline Pendulum')
    axs[0].plot(generate_x_vals(average_rewards), average_rewards, '-r', label="Cumulative Rewards")
    axs[1].plot(generate_x_vals(all_rewards), all_rewards, '-g', label="All rewards")
    axs[2].plot(generate_x_vals(average_losses_critic), average_losses_critic, '*-b', label="AVG Losses Critic")
    axs[2].plot(generate_x_vals(average_losses_actor), average_losses_actor, '-r', label="AVG Losses Actor")
    axs[3].plot(generate_x_vals(all_losses_critic), all_losses_critic, '-b', label="Losses Critic")
    axs[3].plot(generate_x_vals(all_losses_actor), all_losses_actor, '-r', label="Losses Actor")
    axs[0].set_xlabel("Epizody")
    axs[1].set_xlabel("Iteracje")
    axs[2].set_xlabel("Epizody")
    axs[3].set_xlabel("Iteracje")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    axs[3].legend(loc="upper left")

    plt.show()

def run():
    '''
    TODO trezba to przeniesc do pliku experiments.py żeby wykresy były niezależne od modelu
    '''
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


    ac = ActorCriticModel(env = env,
                          num_states=num_states,
                          upper_bound=upper_bound,
                          lower_bound=lower_bound,
                          max_episodes=100,
                          episode_len=3000)

    rewards_history, actor_losses_history, critic_losses_history = ac.learn()

    draw_results(actor_losses_history, critic_losses_history, rewards_history)

run()
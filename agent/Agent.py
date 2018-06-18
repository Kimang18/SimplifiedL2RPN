__author__ = 'kimangkhun'

import random
import numpy as np
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
from collections import deque


class Actor_Critic:
    def __init__(self, state_size, action_shape):
        self.state_size = state_size
        self.action_shape = action_shape
        self.action_size = 2
        self.gamma = 0.95 # discount rate
        self.learning_rate = 0.01
        self.memory = deque(maxlen=2000)
        self.train_fn = []
        self._build_model()

    def _build_model(self):
        state = Input(shape=(self.state_size,), dtype='float32', name='state')
        hiddens = []
        for i in range(self.action_shape):
            hiddens.append(Dense(5, activation='tanh', name='shared{}'.format(i))(state))

        #hidden1 = Dense(32, activation='tanh', name="layer1")(state)

        actions = []
        for i in range(self.action_shape):
            actions.append(Dense(self.action_size, activation='softmax', name='actor{}'.format(i))(hiddens[i]))
        shared = concatenate(hiddens)
        hidden2 = Dense(6, activation='relu', name="layer2")(shared)
        value = Dense(1, activation='linear', name='critic')(hidden2)

        #Actor
        self.actor_models = []
        for i in range(self.action_shape):
            self.actor_models.append(Model(inputs=state, outputs=actions[i]))
            self._build_train_fn(i)

        # Critic
        self.critic_model = Model(inputs=state, outputs=value)
        self.critic_model.compile(
            loss='mse',
            optimizer=Adam(lr=0.1 * self.learning_rate, clipnorm=1.)) #clipvalue=0.5

    def _build_train_fn(self, i):
        act_prob_placeholder = self.actor_models[i].output
        act_onehot_placeholder = K.placeholder(shape=(self.action_size,),
                                               name="action_onehot")
        td_error_placeholder = K.placeholder(shape=(1,1),
                                 name="td_error")
        action_prob = K.sum(act_prob_placeholder*act_onehot_placeholder, axis=1)
        log_action_prob = K.log(K.clip(action_prob, 1e-10, 1.0))
        loss = (-log_action_prob) * td_error_placeholder
        loss = K.mean(loss)
        adam = Adam(lr=0.1 * self.learning_rate) #clipvalue=0.5
        updates = adam.get_updates(params=self.actor_models[i].trainable_weights,
                                   loss=loss)

        self.train_fn.append(K.function(inputs=[self.actor_models[i].input,
                                           act_onehot_placeholder,
                                           td_error_placeholder],
                                   outputs=[],
                                   updates=updates))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.learn_step(state, action, reward, next_state, done)

    def choose_action(self, state):
        #print("nan : {}".format(np.isnan(state).any()))
        #print("inf : {}".format(np.isinf(state).any()))
        #if np.random.rand() <= self.epsilon:
            #return np.random.random_integers(0, 1, size=self.action_shape)
            #return np.ones(self.action_shape, dtype=int)
        state = np.reshape(state, [1, self.state_size])
        action = np.ones(self.action_shape)
        for i in range(self.action_shape):
            act_prob = self.actor_models[i].predict(state)[0]
            if not np.isnan(act_prob).any():
                action[i] = np.random.choice(range(self.action_size), p=act_prob)
            else:
                print("prob_nan!")
                raise ValueError
        return action

    def learn_step(self, state, action, reward, next_state, done):
        if next_state is not None:
            next_state = np.reshape(next_state, [1,self.state_size])
        state = np.reshape(state, [1, self.state_size])
        td_target = reward * np.ones((1,1))
        if not done:
            if(next_state is None):
                print("next_state is none")
            else:
                pred = self.critic_model.predict(next_state)
                td_target = td_target + self.gamma * pred
        td_error = td_target - self.critic_model.predict(state)
        self.critic_model.fit(state, td_target, epochs=1, verbose=0)

        for i in range(self.action_shape):
            action_onehot = np_utils.to_categorical(action[i], num_classes=self.action_size)
            assert state.shape[1] == self.state_size, "{} != {}".format(state.shape[1], self.state_size)
            assert len(action_onehot) == self.action_size, "{} != {}".format(len(action_onehot), self.action_size)
            self.train_fn[i]([state, action_onehot, td_error])

    def load_weights(self, dir):
        for i in range(self.action_shape):
            file = dir + "/actor_weights{}.h5".format(i)
            self.actor_models[i].load_weights(file)
        file = dir + "/critic_weights.h5"
        self.critic_model.load_weights(file)

    def save_weights(self, dir):
        for i in range(self.action_shape):
            file = dir + "/actor_weights{}.h5".format(i)
            self.actor_models[i].save_weights(file)
        file = dir + "/critic_weights.h5"
        self.critic_model.save_weights(file)

class DQN:
    def __init__(self, state_size, action_size):
        self.feature_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95           # discount rate
        self.epsilon = 1.0          # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

        # Create all possible action
        self.action_space = []
        self._str_actions = []
        n_action = 0
        while(n_action < 64):
            a = np.random.random_integers(0, 1, size=self.action_size)
            if str(a) in self._str_actions:
                continue
            else:
                self.action_space.append(a)
                self._str_actions.append(str(a))
                n_action += 1

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.feature_size, activation='tanh'))
        model.add(Dense(64, activation='linear'))   # len(action_space) = 64
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.random_integers(0, 1, size=self.action_size)

        act_values = self.model.predict(state)
        idx_action = np.argmax(act_values[0])
        return self.action_space[idx_action]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                ))
            target_f = self.model.predict(state)
            target_f[0][self._str_actions.index(str(action))] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self, name):
        name = name+".h5"
        self.model.load_weights(name)

    def save_weights(self, name):
        name = name+".h5"
        self.model.save_weights(name)

class DDQN:
    def __init__(self, state_size, action_size):
        self.feature_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95           # discount rate
        self.epsilon = 1.0          # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Create all possible action
        self.action_space = []
        self._str_actions = []
        n_action = 0
        while(n_action < 64):
            a = np.random.random_integers(0, 1, size=self.action_size)
            if str(a) in self._str_actions:
                continue
            else:
                self.action_space.append(a)
                self._str_actions.append(str(a))
                n_action += 1

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.feature_size, activation='tanh'))
        model.add(Dense(64, activation='linear'))   # len(action_space) = 64
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.random_integers(0, 1, size=self.action_size)

        act_values = self.model.predict(state)
        idx_action = np.argmax(act_values[0])
        return self.action_space[idx_action]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.target_model.predict(next_state)[0]
                ))
            target_f = self.model.predict(state)
            target_f[0][self._str_actions.index(str(action))] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self, name):
        name = name+".h5"
        self.model.load_weights(name)

    def save_weights(self, name):
        name = name+".h5"
        self.model.save_weights(name)

class Greedy:
    def __init__(self, action_size):
        self.action_size = action_size
        # Create all possible action
        self.action_space = []
        self._str_actions = []
        n_action = 0
        while (n_action < 64):
            a = np.random.random_integers(0, 1, size=self.action_size)
            if str(a) in self._str_actions:
                continue
            else:
                self.action_space.append(a)
                self._str_actions.append(str(a))
                n_action += 1
    def choose_action(self, state, env):
        best_action = np.ones(env.amt_lines, dtype=int)
        _, reward, done, info = env.simulate(best_action)
        if reward == 1:
            return best_action

        best_reward = -float('inf')
        rewards = []

        for action in self.action_space:
            _, reward, done, info = env.simulate(action)
            rewards.append(reward)
            if best_reward < reward:
                best_action = np.copy(action)
                best_reward = reward
        print(best_action, best_reward)
        return best_action

if __name__=="__main__":
    from environment.game import Environment
    env = Environment()
    env.seed(3)
    s = env.reset()
    n_features = len(s)
    ag = Actor_Critic(n_features, env.amt_lines)
    ag.load_weights("Data")                        # for A2C: load trained weights
    # ag = DQN(n_features, env.amt_lines)
    ag = Greedy(env.amt_lines)
    mobile_returns = deque(maxlen=200)
    mobile_timesteps = deque(maxlen=200)
    # batch_size = 5                                 # for DQN
    avg_returns = []
    avg_timesteps = []
    EPISODES = 1000
    for i_ep in range(EPISODES):
        total_rewards = 0.0
        # s = np.reshape(s, [1, ag.feature_size])    # for DQN
        for h in range(200):
            env.render()
            action = ag.choose_action(s, env)
            #action = np.random.random_integers(0, 1, size=env.amt_lines)
            sprime, reward, done, info = env.step(action)
            # if sprime is not None:                                 #
            #    sprime = np.reshape(sprime, [1, ag.feature_size])  # For DQN
            # ag.remember(s, action, reward, sprime, done)           #
            #ag.learn_step(s, action, reward, sprime, done)     # For actor critic - online learning
            total_rewards += reward
            if done:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()
                break
            elif h == 199:
                print("Epi: \t{}, Ts: \t{}, total rewards: \t{}".format(i_ep, h, total_rewards))
                mobile_returns.append(total_rewards)
                mobile_timesteps.append(h)
                avg_returns.append(np.mean(mobile_returns))
                avg_timesteps.append(np.mean(mobile_timesteps))
                s = env.reset()

            # if len(ag.memory) > batch_size:                     #
            #    ag.replay(batch_size)                           #
        # if i_ep % 10 == 0:                                      # For DQN
            # if ag.epsilon > ag.epsilon_min:                     #
            #    ag.epsilon *= ag.epsilon_decay                  #


    #ag.save_weights("Data/dqn_weight")                         # For DQN
    ag.save_weights("Data")

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_returns)
    plt.ylabel("average return")
    plt.xlabel("number of episodes")
    plt.title("Average return over episodes")
    plt.xlim(0, EPISODES)
    plt.ylim(-250, 250)

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.plot(range(len(avg_returns)), avg_timesteps)
    plt.ylabel("number of timesteps")
    plt.xlabel("number of episodes")
    plt.title("Average timesteps over episodes")
    plt.xlim(0, EPISODES)
    plt.ylim(-250, 250)
    plt.show()
    np.savetxt("Data/average_returns_a2c_inj.txt", avg_returns, fmt='%.3f')
    np.savetxt("Data/average_timesteps_a2c_inj.txt", avg_timesteps, fmt='%.3f')

__author__ = 'kimangkhun'

import random
import numpy as np
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, concatenate, add, LSTM, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D, PReLU, AveragePooling1D
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
from collections import deque

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Action_Space:
    def __init__(self):
        self.nb_lines = 6
        # Create all possible action
        self._actions = []
        self._str_actions = []
        n_action = 0
        np.random.seed(1)
        while (n_action < 64):
            a = np.random.random_integers(0, 1, size=self.nb_lines)
            if str(a) in self._str_actions:
                continue
            else:
                self._actions.append(a)
                self._str_actions.append(str(a))
                n_action += 1

        # Help agent to reduce action space by deleting useless action
        toRemove = []
        for i in range(len(self._actions)):
            if len(np.nonzero(self._actions[i])[0]) < 4:
                toRemove.append(i)
        for i in sorted(toRemove, reverse=True):
            del self._actions[i]
            del self._str_actions[i]
        self.n = len(self._actions)
    def __repr__(self):
        print(self._actions)
    def get(self, i):
        return self._actions[i]
    def get_str(self, i):
        return self._str_actions[i]
    def get_index(self, action):
        return self._str_actions.index(str(action))

class Actor_Critic:
    # TODO: Actor predicts one of 13 available actions
    def __init__(self, obs_shape, action_shape):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = 1.0              # discount rate
        self.learning_rate = 0.0023
        self.memory = deque(maxlen=2048)
        self.tau = 0.999
        self.action_space = Action_Space()
        self.train_fn = []
        self._build_model()
        self.updated = False

    def _build_model(self):
        # Ordinary Actor
        state = Input(shape=self.obs_shape, dtype='float32', name='state')

        # Actor net
        h1 = BatchNormalization()(state)
        h1 = Conv1D(64, 3, activation='relu')(h1)
        h1 = Conv1D(64, 3, activation='relu')(h1)
        h1 = GlobalAveragePooling1D()(h1)
        h1 = Dropout(0.5)(h1)

        hidden1 = Dense(64, activation='tanh', name='state1')(h1)
        policy = Dense(self.action_space.n, activation='softmax', name='actor')(hidden1)
        self.actor_model = Model(inputs=state, outputs=policy)
        self._build_train_act()

        # Critic
        value = Dense(1, activation='linear', name='critic')(h1)
        self.critic_model = Model(inputs=state, outputs=value)
        self.critic_target_model = Model(inputs=state, outputs=value)
        self.critic_model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        self.critic_target_model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

    def _build_train_act(self):
        act_prob_placeholder = self.actor_model.output
        act_onehot_placeholder = K.placeholder(shape=(self.action_space.n,),
                                               name='action_onehot')
        td_error_placeholder = K.placeholder(shape=(1, 1),
                                             name='td_error')
        action_prob = K.sum(K.clip(act_prob_placeholder*act_onehot_placeholder, 1e-10, 1.0), axis=1)
        log_action_prob = K.log(action_prob)    #K.clip(action_prob, 1e-10, 1.0))
        loss = (-log_action_prob) * td_error_placeholder
        loss = K.mean(loss)
        adam = Adam(lr=0.001)
        updates = adam.get_updates(params=self.actor_model.trainable_weights,
                                   loss=loss)
        self.train_fn_act = K.function(inputs=[self.actor_model.input,
                                       act_onehot_placeholder,
                                       td_error_placeholder],
                                       outputs=[],
                                       updates=updates)

    def _update_target_model(self):
        if not self.updated:
            self.critic_target_model.set_weights(self.critic_model.get_weights())
            self.updated = True
        else:
            behavior_weights = self.critic_model.get_weights()
            target_weights = self.critic_target_model.get_weights()
            new_weights = []
            for i in range(len(behavior_weights)):
                # Polyak Average
                new_weights.append(self.tau * target_weights[i] +
                                   (1 - self.tau) * behavior_weights[i])
            self.critic_target_model.set_weights(new_weights)

    def _build_train_fn(self, i):
        act_prob_placeholder = self.actor_model[i].output
        act_onehot_placeholder = K.placeholder(shape=(self.action_space.n,),
                                               name="action_onehot")
        td_error_placeholder = K.placeholder(shape=(1,),
                                 name="td_error")
        action_prob = K.sum(act_prob_placeholder*act_onehot_placeholder, axis=1)
        log_action_prob = K.log(K.clip(action_prob, 1e-10, 1.0))
        loss = (-log_action_prob) * td_error_placeholder
        loss = K.mean(loss)
        adam = Adam(lr=0.1 * self.learning_rate)
        updates = adam.get_updates(params=self.actor_model[i].trainable_weights,
                                   loss=loss)

        self.train_fn.append(K.function(inputs=[self.actor_model[i].input,
                                           act_onehot_placeholder,
                                           td_error_placeholder],
                                   outputs=[],
                                   updates=updates))

    def remember(self, state, action, reward, next_state, done):
        obs = np.reshape(state, [1, 15, 2])
        next_obs = np.reshape(next_state, [1, 15, 2])

        self.memory.append((obs, action, reward, next_obs, done))

    def store_transition(self, state, action, reward):
        # For MC method
        state = np.reshape(state, [1, 15, 2])
        self.state_buf.append(state)
        self.action_buf.append(action)
        self.reward_buf.append(reward)

    def learn(self):
        #TODO: refacto
        # For Monte Carlo method
        n = len(self.state_buf)
        v = []
        for i in range(n):
            v.append(self.critic_model.predict(self.state_buf[i]))

        v_next = v[1:n]

        q_sa = []
        for i in range(len(v_next)):
            q_sa.append(self.reward_buf[i] + self.gamma * v_next[i])
        q_sa.append(self.reward_buf[n-1])

        avantages = []
        for i in range(n):
            avantages.append(q_sa[i] - v[i])

        # train Critic
        for i in range(n):
            td_target = np.reshape(q_sa[i], [1, 1])
            self.critic_model.fit(self.state_buf[i], td_target, epochs=1, verbose=0)

        # train Actors
        for i in range(n):
            state = self.state_buf[i]

            action_onehot = np_utils.to_categorical(self.action_buf[i][0], num_classes=self.action_size)
            self.train_fn[0]([state, action_onehot, avantages[i]])

            for j in range(1, self.action_shape):
                action_onehot = np_utils.to_categorical(self.action_buf[i][j], num_classes=self.action_size)
                state = np.append(state, self.action_buf[i][j-1])
                state = np.reshape(state, [1, state.shape[0]])
                self.train_fn[j]([state, action_onehot, avantages[i]])
            """
            action_onehot = np_utils.to_categorical(self.action_space.get_index(self.action_buf[i]),
                                                    num_classes=self.action_space.n)
            self.train_fn_act([state, action_onehot, avantages[i]])
            """

        # Clean trajectory
        self.state_buf.clear()
        self.action_buf.clear()
        self.reward_buf.clear()

    def replay(self, batch_size):

        s_batch = []
        a_onehot_batch = []
        td_batch = []
        td_error_batch = []
        a_res_batch = []

        for state, action, reward, next_state, done in self.memory:
            #self.learn_step(state, action, reward, next_state, done)

            action_reshaped = np.reshape(action, [1, self.action_shape])
            td_target = reward * np.ones((1, 1))
            if not done:
                # Get the value from critic_target
                pred = self.critic_target_model.predict(next_state)
                td_target = td_target + self.gamma * pred

            action_onehot = np_utils.to_categorical(self.action_space.get_index(action),
                                                    num_classes=self.action_space.n)
            if len(a_onehot_batch) == 0:
                a_res_batch = np.copy(action_reshaped)
                s_batch = np.copy(state)
                td_batch = np.copy(td_target)
                a_onehot_batch = np.copy(action_onehot)
            else:
                a_res_batch = np.vstack((a_res_batch, action_reshaped))
                s_batch = np.vstack((s_batch, state))
                td_batch = np.vstack((td_batch, td_target))
                a_onehot_batch = np.vstack((a_onehot_batch, action_onehot))

        self.critic_model.fit(s_batch, td_batch, epochs=1, verbose=0)
        td_target = self.critic_model.predict(s_batch)

        # Evaluate the policy
        for i in range(td_target.shape[0]-1):
            td_error = td_target[i+1] - td_target[i]
            if len(td_error_batch) == 0:
                td_error_batch = np.copy(td_error)
            else:
                td_error_batch = np.vstack((td_error_batch, td_error))

        # For the last state (i == td_target.shape[0] - 1)
        if td_target.shape[0] == 1:
            reward = self.memory[0][2]
            td_error_batch = reward * np.ones((1, 1)) - td_target[0]
        else:
            td_error_batch = np.vstack((td_error_batch, np.zeros((1, 1))))
            #td_error_batch = (td_error_batch - np.mean(td_error_batch))/np.std(td_error_batch)
        self.train_fn_act([s_batch, a_onehot_batch, td_error_batch])
        self.memory.clear()

    def prob_action(self, state):
        obs = np.reshape(state, [1, 15, 2])
        return self.actor_model.predict(obs)[0]

    def choose_action(self, state):
        #state_copy = np.reshape(state, [1, self.state_size])
        obs = np.reshape(state, [1, 15, 2])

        ### When there is only 1 actor
        act_prob = self.actor_model.predict(obs)[0]
        if not np.isnan(act_prob).any():
            act_chosen = np.random.choice(range(self.action_space.n), p=act_prob)
            return self.action_space.get(act_chosen)
        else:
            print("prob_nan!")
            raise ValueError

    def learn_step(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, 15, 2])
        next_state = np.reshape(next_state, [1, 15, 2])
        td_target = reward * np.ones((1, 1))
        if not done:
            # Get the value from critic_target
            pred = self.critic_model.predict(next_state)
            td_target = td_target + self.gamma * pred

        self.critic_model.fit(state, td_target, epochs=1, verbose=0)

        # Evaluate action
        td_target = reward * np.ones((1, 1))
        if not done:
            pred = self.critic_model.predict(next_state)
            td_target = td_target + self.gamma * pred
        td_error = td_target - self.critic_model.predict(state)

        action_onehot = np_utils.to_categorical(self.action_space.get_index(action), num_classes=self.action_space.n)
        self.train_fn_act([state, action_onehot, td_error])

    def load_weights(self, dir):
        """
        for i in range(self.action_shape):
            file = dir + "/actor_weights{}.h5".format(i)
            self.actor_models[i].load_weights(file)
        """
        self.actor_model.load_weights(dir + "/actor_weight.h5")

        file = dir + "/critic_weights.h5"
        self.critic_model.load_weights(file)
        self._update_target_model()

    def save_weights(self, dir):
        """
        for i in range(self.action_shape):
            file = dir + "/actor_weights{}.h5".format(i)
            self.actor_models[i].save_weights(file)
        """
        self.actor_model.save_weights(dir + "/actor_weight.h5")

        file = dir + "/critic_weights.h5"
        self.critic_model.save_weights(file)


class DQN:
    def __init__(self, obs_shape):
        self.obs_shape = obs_shape
        self.memory = deque(maxlen=2048)
        self.gamma = 1.0            # discount rate
        self.epsilon = 1.0          # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0023
        self.action_space = Action_Space()
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.tau = 0.999
        self.updated = False

    def update_target_model(self):
        if not self.updated:
            self.target_model.set_weights(self.model.get_weights())
            self.updated = True
        else:
            behavior_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            new_weights = []
            for i in range(len(behavior_weights)):
                # Polyak Average
                new_weights.append(self.tau * target_weights[i] +
                                   (1 - self.tau) * behavior_weights[i])
            self.target_model.set_weights(new_weights)

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.obs_shape))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        """
        model.add(LSTM(32, return_sequences=True, input_shape=self.obs_shape))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        """
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        obs = np.reshape(state, [1, 15, 2])
        next_obs = np.reshape(next_state, [1, 15, 2])
        if reward > 0:
            self.memory.append((obs, action, reward, next_obs, done))
        self.memory.append((obs, action, reward, next_obs, done))

    def choose_action(self, state):
        obs = np.reshape(state, [1, 15, 2])
        #obs = np.reshape(state, [1, 2, 15])
        """
        act_values = self.model.predict(obs)[0]
        act_prob = softmax(act_values)
        idx_action = np.random.choice(range(self.action_space.n), p=act_prob)
        """
        if np.random.rand() <= self.epsilon:
            return self.action_space.get(np.random.randint(0, self.action_space.n))
        act_values = self.model.predict(obs)
        idx_action = np.argmax(act_values[0])
        return self.action_space.get(idx_action)

    def quality(self, state):
        obs = np.reshape(state, [1, 15, 2])
        act_values = self.model.predict(obs)
        return softmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        s_batch = []
        target_batch = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.target_model.predict(next_state)[0]
                ))
            target_f = self.model.predict(state)
            target_f[0][self.action_space.get_index(action)] = target
            if len(s_batch) == 0:
                s_batch = np.copy(state)
                target_batch = np.copy(target_f)
            else:
                s_batch = np.vstack((s_batch, state))
                target_batch = np.vstack((target_batch, target_f))
        self.model.fit(s_batch, target_batch, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self, name):
        name = name+".h5"
        self.model.load_weights(name)
        self.update_target_model()

    def save_weights(self, name):
        name = name+".h5"
        self.target_model.save_weights(name)


class DDQN:
    def __init__(self, obs_shape, action_size):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2048)
        self.gamma = 1.0          # discount rate
        self.epsilon = 1.0        # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0023
        self.tau = 0.999              #soft update between behavior and target model
        self.action_space = Action_Space()
        self.behavior_model = self._build_model()
        self.target_model = self._build_model()
        self.updated = False

    def update_target_model(self):
        if not self.updated:
            self.target_model.set_weights(self.behavior_model.get_weights())
            self.updated = True
        else:
            behavior_weights = self.behavior_model.get_weights()
            target_weights = self.target_model.get_weights()
            new_weights = []
            for i in range(len(behavior_weights)):
                # Polyak Average
                new_weights.append(self.tau * target_weights[i] +
                                   (1 - self.tau) * behavior_weights[i])
            self.target_model.set_weights(new_weights)

    def _huber_loss(self, y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.obs_shape))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        """
        model.add(LSTM(32, return_sequences=True, input_shape=self.obs_shape))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        """
        model.add(Dense(self.action_space.n, activation='linear'))  # len(action_space) = 64
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model
        """
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.observation_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model
        """

    def remember(self, state, action, reward, next_state, done):
        obs = np.reshape(state, [1, 15, 2])
        # obs = np.reshape(state, [1, 2, 15]) # when using LSTM
        next_obs = np.reshape(next_state, [1, 15, 2])
        if reward > 0:
            self.memory.append((obs, action, reward, next_obs, done))
        self.memory.append((obs, action, reward, next_obs, done))

    def choose_action(self, state):
        obs = np.reshape(state, [1, 15, 2])
        """
        act_values = self.behavior_model.predict(obs)[0]
        act_prob = softmax(act_values)
        idx_action = np.random.choice(range(self.action_space.n), p=act_prob)
        """
        if np.random.rand() <= self.epsilon:
            return self.action_space.get(np.random.randint(0, self.action_space.n))
        act_values = self.behavior_model.predict(obs)
        idx_action = np.argmax(act_values[0])

        return self.action_space.get(idx_action)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        s_batch = []
        target_batch = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                current_q = self.behavior_model.predict(next_state)[0]
                target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(current_q)]
            target_f = self.behavior_model.predict(state)
            target_f[0][self.action_space.get_index(action)] = target
            if len(s_batch) == 0:
                s_batch = np.copy(state)
                target_batch = np.copy(target_f)
            else:
                s_batch = np.vstack((s_batch, state))
                target_batch = np.vstack((target_batch, target_f))
        self.behavior_model.fit(s_batch, target_batch, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def quality(self, state):
        obs = np.reshape(state, [1, 15, 2])
        act_values = self.behavior_model.predict(obs)
        return softmax(act_values[0])

    def load_weights(self, name):
        name = name+".h5"
        self.behavior_model.load_weights(name)
        self.update_target_model()

    def save_weights(self, name):
        name = name+".h5"
        self.behavior_model.save_weights(name)


class Greedy:
    def __init__(self):
        # Create all possible action
        self.action_space = Action_Space()
    def choose_action(self, state, env):
        best_action = self.exhautive_search(state, env)
        return best_action, self.action_space.get_index(best_action)

    def first_good(self, state, env):
        overflow = state[14][1] > 0
        if overflow:
            env_action = np.zeros(17)
            n_act = env.amt_lines
            action = np.ones(env.amt_lines, dtype=int)
            env_action[11:] = np.copy(action)
            _, reward, done, info = env.simulate(env_action)
            if reward >= 0.0:
                return action
            for i in range(n_act):
                action = np.ones(n_act, dtype=int)
                action[i] = 0
                env_action[11:] = np.copy(action)
                _, reward, done, info = env.simulate(env_action)
                if reward >= 0.0:
                    return action
            for i in range(n_act-1):
                for j in range(i+1, n_act):
                    action = np.ones(n_act, dtype=int)
                    action[i] = 0
                    action[j] = 0
                    env_action[11:] = np.copy(action)
                    _, reward, done, info = env.simulate(env_action)
                    if reward >= 0.0:
                        return action
        else:
            env_action = np.zeros(17)
            n_act = env.amt_lines
            action = np.ones(env.amt_lines, dtype=int)
            env_action[11:] = np.copy(action)
            _, reward, done, info = env.simulate(env_action)
            if reward >= 0.0:
                return action
            for i in range(n_act):
                action = np.ones(n_act, dtype=int)
                action[i] = 0
                env_action[11:] = np.copy(action)
                _, reward, done, info = env.simulate(env_action)
                if reward >= 0.0:
                    return action
            for i in range(n_act - 1):
                for j in range(i + 1, n_act):
                    action = np.ones(n_act, dtype=int)
                    action[i] = 0
                    action[j] = 0
                    env_action[11:] = np.copy(action)
                    _, reward, done, info = env.simulate(env_action)
                    if reward >= 0.0:
                        return action
        return np.ones(env.amt_lines, dtype=int)

    def exhautive_search(self, state, env):
        best_reward = -2
        for i in range(self.action_space.n):
            env_action = np.zeros(17)
            action = self.action_space.get(i)
            env_action[11:] = np.copy(action)
            _, reward, done, info = env.simulate(env_action)
            if best_reward < reward:
                best_reward = reward
                best_action = np.copy(action)
        return best_action


class Greedy_Prior(object):
    def __init__(self):
        """ Greedy Prior implemented similarly to MCTS
            We search to disconnect a line using prior knowledge on how many times the lines
            help saving the overflows.
            For simplicity, we disconnect line {1, 2, 3, 4} only
        """
        self._w = np.ones(5, dtype=float)  # number of saving for the line_i
        self._n = np.ones(5, dtype=float)  # number of simulations for the line_i
        self._N = 0.0  # number of total simulations
        self._c = 1.4142  # exploration parameter
        self._value = np.ones(5, dtype=float) # initial proability

    def choose_action(self, state, env):
        # choose among the four lines
        tested = []
        for i in range(5):
            value_chosen = np.max(np.delete(self._value, tested))
            for i in range(5):
                if value_chosen == self._value[i]:
                    idx_chosen = i
                    break
            self._n[idx_chosen] += 1
            self._N += 1.0
            env_action = np.zeros(17)
            action = np.ones(6, dtype=int)
            action[idx_chosen] = 0
            env_action[11:] = np.copy(action)
            _, reward, done, info = env.simulate(env_action)
            if reward > 0.0:
                self._w[idx_chosen] += 1
                for i in range(len(self._value)):
                    self._value[i] = self._w[i] / self._n[i] \
                                          + self._c * np.sqrt(np.log(self._N) / self._n[i])
                return action
            else:
                for i in range(len(self._value)):
                    self._value[i] = self._w[i] / self._n[i] \
                                          + self._c * np.sqrt(np.log(self._N) / self._n[i])
                tested.append(idx_chosen)
        # The 4 lines disconnection cannot save the overflows
        action = np.ones(6, dtype=int)
        idx_chosen = np.random.choice(range(2), p=0.5*np.ones(2))
        if idx_chosen == 0:
            action[5] = 0
        return action


class mcts(object):
    def __init__(self):
        """ Greedy Prior implemented similarly to MCTS
            We search to disconnect a line using prior knowledge on how many times the lines
            help saving the overflows.
            For simplicity, we disconnect line {1, 2, 3, 4} only
        """
        self._w = np.ones(6, dtype=float)  # number of saving for the line_i
        self._n = np.zeros(6, dtype=float)  # number of simulations for the line_i
        self._N = 0  # number of total simulations
        self._c = 1.4142  # exploration parameter
        self._value = np.ones(6, dtype=float) # initial proability

    def choose_action(self, state, env):
        # donothing first
        env_action = np.zeros(17)
        action = np.ones(6, dtype=int)
        env_action[11:] = action
        _, reward, done, info = env.simulate(env_action)
        if reward > 0.0:
            return action

        # choose among the four lines
        for i in range(6):
            idx_chosen = np.argmax(self._value)
            # update visit counters
            self._n[idx_chosen] += 1
            self._N += 1

            action = np.ones(6, dtype=int)
            action[idx_chosen] = 0
            env_action[11:] = np.copy(action)
            _, reward, done, info = env.simulate(env_action)
            if reward > 0.0:
                self._w[idx_chosen] += 1
                self._value[idx_chosen] = self._w[idx_chosen] / self._n[idx_chosen] \
                                         + self._c * np.sqrt(np.log(self._N) / self._n[idx_chosen])
                return action
            else:
                self._value[idx_chosen] = self._w[idx_chosen] / self._n[idx_chosen] \
                                          + self._c * np.sqrt(np.log(self._N) / self._n[idx_chosen])

                # after update search value, should we go deeper?
                iidx_chosen = np.argmax(self._value)
                if iidx_chosen == idx_chosen:   # search deeper
                    for j in range(i, 6):
                        jdx_chosen = np.argmax(np.delete(self._value, idx_chosen))
                        if jdx_chosen >= idx_chosen:
                            # shift one forward to get exact index
                            jdx_chosen += 1
                        # update visit counters
                        self._n[idx_chosen] += 1
                        self._n[jdx_chosen] += 1
                        self._N += 1

                        action[jdx_chosen] = 0
                        env_action[11:] = np.copy(action)
                        _, reward, done, info = env.simulate(env_action)
                        if reward > 0.0:
                            self._w[idx_chosen] += 1
                            self._w[jdx_chosen] += 1
                            self._value[idx_chosen] = self._w[idx_chosen] / self._n[idx_chosen] \
                                                      + self._c * np.sqrt(np.log(self._N) / self._n[idx_chosen])
                            self._value[jdx_chosen] = self._w[jdx_chosen] / self._n[jdx_chosen] \
                                                      + self._c * np.sqrt(np.log(self._N) / self._n[jdx_chosen])
                            return action
                        else:
                            self._value[idx_chosen] = self._w[idx_chosen] / self._n[idx_chosen] \
                                                      + self._c * np.sqrt(np.log(self._N) / self._n[idx_chosen])
                            self._value[jdx_chosen] = self._w[jdx_chosen] / self._n[jdx_chosen] \
                                                      + self._c * np.sqrt(np.log(self._N) / self._n[jdx_chosen])

class DAgger(object):
    def __init__(self, obs_shape):
        self.obs_shape = obs_shape
        self.action_shape = 6
        self.learning_rate = 0.009
        self.human = Greedy()
        self.action_size = self.human.action_space.n

        self.agent = self._build_model()
        self.memory = deque(maxlen=8000)
        self.met_states = {}

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.obs_shape))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.25))
        """
        model.add(LSTM(32, return_sequences=True, input_shape=self.obs_shape))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        """

        model.add(Dense(self.action_shape, activation='sigmoid'))  # len(action_space) = 64
        model.compile(
            loss=binary_crossentropy,
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, state, action, warmup=True):
        if warmup:
            encoded_state = hash(state.tostring())
            if self.met_states.get(encoded_state) is None:
                self.met_states[encoded_state] = state
                self.memory.append((state, action))
        else:
            self.memory.append((state, action))

    def choose_action(self, state, env):
        encoded_state = hash(state.tostring())
        if self.met_states.get(encoded_state) is None:
            ### Never meet this state before
            self.met_states[encoded_state] = state
            """Human action"""
            action, _ = self.human.choose_action(state, env)

            """Save the human action"""
            self.remember(state, action, False)

        """Agent acts"""
        obs = np.reshape(state, [1, 15, 2])
        act_values = self.agent.predict(obs)[0]
        action = np.ones(env.amt_lines, dtype=int)
        for i in range(len(act_values)):
            if act_values[i] < 0.5:
                action[i] = 0
        return action

    def train(self):
        states = np.asarray([e[0] for e in self.memory])
        actions = np.asarray([e[1] for e in self.memory])
        H = self.agent.fit(states, actions, batch_size=256, epochs=10, verbose=0)
        #print(H.history)

    def save_weights(self, name):
        name = name+".h5"
        self.agent.save_weights(name)

    def load_weights(self, name):
        name = name + ".h5"
        self.agent.load_weights(name)

if __name__=="__main__":
    from environment.game import Environment
    env = Environment()
    env.seed(3)
    s = env.reset()
    n_features = len(s)
    ag = Actor_Critic(n_features, env.amt_lines)
    ag.load_weights("Data")                        # for A2C: load trained weights
    ag = DQN(n_features, env.amt_lines)
    #ag = Greedy(env.amt_lines)
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
            action = ag.choose_action(s)
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
    #plt.ylim(-250, 250)

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

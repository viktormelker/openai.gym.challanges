import os
import random
from collections import deque

import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class BasePolicy:
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states

    def update(self, states, actions, reward, **kwargs):
        raise NotImplementedError

    def get_action(self, state, **kwargs):
        raise NotImplementedError


class SimplePolicy(BasePolicy):
    def __init__(self, num_actions, num_states):
        super().__init__(num_actions, num_states)
        self.learning_rate = 0.1
        self.age_factor = 0.9
        self.action_probabilities = np.array(
            num_states * [num_actions * [0.25]])

    def update(self, states, actions, reward):
        if reward == 0:
            return

        age = len(states) - 1
        for state, action in zip(states, actions):
            self.action_probabilities[state, action] = max(
                0, (self.action_probabilities[state, action] +
                    self.learning_rate * reward * pow(self.age_factor, age)))

            self._normalize(state)
            age -= 1

    def _normalize(self, state):
        self.action_probabilities[state] = (
            self.action_probabilities[state] /
            self.action_probabilities[state].sum())

    def get_action(self, state, attempt=1):
        return np.random.choice(
            self.num_actions, 1, p=self.action_probabilities[state])[0]


class QTablePolicy(BasePolicy):
    def __init__(self, num_actions, num_states):
        super().__init__(num_actions, num_states)
        self.Q = np.zeros([num_states, num_actions])
        self.y = .99
        self.learning_rate = 0.85

    def update(self, states, actions, reward, result_state, **kwargs):
        state = states[-1]
        action = actions[-1]

        # Update Q-Table with new knowledge from last step
        self.Q[state, action] = (
            (1 - self.learning_rate) * self.Q[state, action] +
            self.learning_rate *
            (reward + self.y * np.max(self.Q[result_state, :])))

        return

    def get_action(self, state, attempt, **kwargs):
        # Choose an action by greedily (with noise) picking from Q table
        return (
            np.argmax(self.Q[state, :] + np.random.randn(1, self.num_actions) *
                      (1. / (attempt + 1))))


# Deep Q-learning Agent
class DQNAgent:
    def __init__(
        self, state_size, action_size, gamma=0.95, epsilon=1.0,
        epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
        model_file=None, weight_file=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model_file = model_file
        self.weight_file = weight_file

        if self.model_file is None:
            self.model = self._build_model()
        else:
            self.model = self.load_model()

        if self.weight_file is not None:
            self.load_weights()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        if batch_size > len(self.memory):
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        pass

    def load_model(self):
        return None

    def save_weights(self):
        print('Saved weights to file: ' + self.weight_file)
        self.model.save_weights(self.weight_file)

    def load_weights(self):
        if os.path.exists(self.weight_file):
            self.model.load_weights(self.weight_file)
            print('Loaded weights from file: ' + self.weight_file)
        else:
            print('Could not load weights from non-existing file: ' + self.weight_file)


class DoubleDQNAgent(DQNAgent):
    def __init__(
        self, state_size, action_size, **kwargs
    ):
        super().__init__(
            state_size, action_size, **kwargs)
        self.tau = 0.05
        self.target_model = self._build_model()

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def replay(self, batch_size=32):
        if batch_size > len(self.memory):
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.target_model.predict(next_state)[0])
            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

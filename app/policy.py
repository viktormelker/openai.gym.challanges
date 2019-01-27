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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

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

    def replay(self, batch_size):
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

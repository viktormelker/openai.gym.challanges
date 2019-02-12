import os
import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class QLearningPolicy:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

    def save_transition(self, state, action, reward, next_state, done, **kwargs):
        raise NotImplementedError

    def get_action(self, state, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class QTablePolicy(QLearningPolicy):
    def __init__(self, state_size, action_size):
        super().__init__(state_size=state_size, action_size=action_size)
        self.Q = np.zeros([state_size, action_size])
        self.y = 0.99
        self.learning_rate = 0.85

    def save_transition(self, state, action, reward, next_state, done, **kwargs):
        self.Q[state, action] = (1 - self.learning_rate) * self.Q[
            state, action
        ] + self.learning_rate * (reward + self.y * np.max(self.Q[next_state, :]))

    def get_action(self, state, attempt, **kwargs):
        # Choose an action by greedily (with noise) picking from Q table
        return np.argmax(
            self.Q[state, :]
            + np.random.randn(1, self.action_size) * (1.0 / (attempt + 1))
        )

    def train(self, **kwargs):
        # training is automatically done in the save_transition step
        pass


class RandomPolicy(QLearningPolicy):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def save_transition(self, state, action, reward, next_state, done, **kwargs):
        pass

    def get_action(self, state, **kwargs):
        return np.argmax(np.random.randn(1, self.action_size))

    def train(self, **kwargs):
        pass


# Deep Q-learning Agent
class DQNAgent(QLearningPolicy):
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        model_file=None,
        weight_file=None,
        potential_function=lambda x: 0,
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
        self.potential_function = potential_function

        if self.model_file is None:
            self.model = self._build_model()
        else:
            self.model = self.load_model()

        if self.weight_file is not None:
            self.load_weights()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, **kwargs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def _replay(self, batch_size=32):
        if batch_size > len(self.memory):
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = (
                    reward
                    + (
                        -self.potential_function(state)
                        + self.gamma * self.potential_function(next_state)
                    )
                    + self.gamma * np.amax(self.model.predict(next_state)[0])
                )

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
        print("Saved weights to file: " + self.weight_file)
        self.model.save_weights(self.weight_file)

    def load_weights(self):
        if os.path.exists(self.weight_file):
            self.model.load_weights(self.weight_file)
            print("Loaded weights from file: " + self.weight_file)
        else:
            print("Could not load weights from non-existing file: " + self.weight_file)

    def save_transition(self, state, action, reward, next_state, done, **kwargs):
        self._remember(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

    def train(self, **kwargs):
        self._replay(**kwargs)


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.tau = 0.05
        self.target_model = self._build_model()

    def _target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (
                1 - self.tau
            )

        self.target_model.set_weights(target_weights)

    def _replay(self, batch_size=32):
        if batch_size > len(self.memory):
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = (
                    reward
                    - self.potential_function(state)
                    + self.gamma * self.potential_function(next_state)
                    + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                )
            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_transition(self, state, action, reward, next_state, done, **kwargs):
        self._remember(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

    def train(self, **kwargs):
        self._replay(**kwargs)
        self._target_train()

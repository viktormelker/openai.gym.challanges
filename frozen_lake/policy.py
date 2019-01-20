import numpy as np


class Policy:

    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = 0.1
        self.age_factor = 0.9

    action_probabilities = np.array(16 * [4 * [0.25]])

    def update(self, states, actions, reward):
        age = len(states) - 1
        for state, action in zip(states, actions):
            self.action_probabilities[state, action] = max(0, (
                self.action_probabilities[state, action] +
                self.learning_rate * reward * pow(self.age_factor, age)
            ))

            self._normalize(state)
            age -= 1

    def _normalize(self, state):
        self.action_probabilities[state] = (
            self.action_probabilities[state] / self.action_probabilities[state].sum())

    def get_action(self, state):
        return np.random.choice(
            self.num_actions, 1, p=self.action_probabilities[state]
        )[0]

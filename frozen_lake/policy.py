import numpy as np


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
            self.action_probabilities[state, action] = max(0, (
                self.action_probabilities[state, action] +
                self.learning_rate * reward * pow(self.age_factor, age)
            ))

            self._normalize(state)
            age -= 1

    def _normalize(self, state):
        self.action_probabilities[state] = (
            self.action_probabilities[state] / self.action_probabilities[state].sum())

    def get_action(self, state, attempt=1):
        return np.random.choice(
            self.num_actions, 1, p=self.action_probabilities[state]
        )[0]


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
            self.Q[state, action] + self.learning_rate *
            (reward + self.y * np.max(self.Q[result_state, :]) - self.Q[state, action])
        )

    def get_action(self, state, attempt, **kwargs):
        # Choose an action by greedily (with noise) picking from Q table
        return (np.argmax(self.Q[state, :] + np.random.randn(1, self.num_actions) * (1./(attempt + 1))))

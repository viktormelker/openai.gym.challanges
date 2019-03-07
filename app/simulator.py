from collections import namedtuple

import numpy as np
from policies.q_learning import QLearningPolicy

SimulationData = namedtuple("SimulatonData", ["state", "next_state", "reward", "done"])


class QLearningSimulator:
    state_slice = slice(0, 1, 1)
    next_state_slice = slice(1, 2, 1)
    reward_slice = slice(2, 3, 1)
    done_slice = slice(3, 4, 1)

    def __init__(
        self,
        policy: QLearningPolicy,
        environment,
        state_size,
        render_gui=False,
        num_episodes=50,
        max_time=500,
        **kwargs
    ):
        self.policy = policy
        self.environment = environment
        self.state_size = state_size
        self.render_gui = render_gui
        self.num_episodes = num_episodes
        self.max_time = max_time

    def simulate(self):
        for self.episode in range(self.num_episodes):
            state = self.environment.reset()
            state = np.reshape(state, [1, self.state_size])

            self.simulation_data = []

            for self.time_t in range(self.max_time):

                if self.render_gui:
                    self.environment.render()

                action = self.policy.get_action(state)

                next_state, reward, done, _ = self.environment.step(action)

                self.simulation_data.append(
                    SimulationData(state, next_state, reward, done)
                )

                next_state = np.reshape(next_state, [1, self.state_size])

                self.policy.save_transition(state, action, reward, next_state, done)
                self.policy.train()

                state = next_state

                if done:
                    self.on_episode_done()
                    break

            self.on_episode_ended()

            if self.is_finished():
                break

        self.on_training_finished()

    def on_episode_done(self):
        pass

    def on_episode_ended(self):
        pass

    def is_finished(self):
        return False

    def on_training_finished(self):
        pass

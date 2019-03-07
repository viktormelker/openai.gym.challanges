from collections import deque

from app.simulator import QLearningSimulator


class MountainCarSimulator(QLearningSimulator):
    best_positions = deque(maxlen=40)

    def __init__(self, success_x_pos, **kwargs):
        super().__init__(**kwargs)
        self.success_x_pos = success_x_pos

    def on_training_finished(self):
        self.policy.save_weights()

    def on_episode_done(self):
        rewards = [data.reward for data in self.simulation_data]
        positions = [data.state[0, 0] for data in self.simulation_data]
        self.best_positions.append(max(positions))
        average_best_pos = sum(self.best_positions) / len(self.best_positions)

        print(
            "episode: {0:4d}/{1:4d}, time_t: {2:3d}, max pos: {3:6.2f}, avg pos: {4:6.2f}, total reward: {5:6.2f}".format(
                self.episode,
                self.num_episodes,
                self.time_t,
                max(positions),
                average_best_pos,
                sum(rewards),
            )
        )

    def is_finished(self):
        average_best_pos = sum(self.best_positions) / len(self.best_positions)

        if average_best_pos > self.success_x_pos:
            print(
                "Successfully finished the challenge in {} training runs!".format(
                    self.episode
                )
            )
            return True
        else:
            return False

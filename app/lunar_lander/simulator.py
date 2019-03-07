import itertools
from collections import deque

from app.simulator import QLearningSimulator


class LunarLanderSimulator(QLearningSimulator):
    total_rewards = deque(maxlen=1000)
    averaging_length = 40

    def __init__(self, target_average_reward, **kwargs):
        super().__init__(**kwargs)
        self.target_average_reward = target_average_reward

    def on_training_finished(self):
        self.policy.save_weights()

    def on_episode_ended(self):
        rewards = [data.reward for data in self.simulation_data]
        self.total_rewards.append(sum(rewards))
        average_total_reward = self.get_average_total_reward()

        print(
            "episode: {0:4d}/{1:4d}, time_t: {2:3d}, reward: {3:8.2f}, avg reward: {4:8.2f}".format(
                self.episode,
                self.num_episodes,
                self.time_t,
                self.total_rewards[-1],
                average_total_reward,
            )
        )

    def is_finished(self):
        average_total_reward = self.get_average_total_reward()

        if average_total_reward > self.target_average_reward:
            print(
                "Successfully finished the challenge in {} training runs!".format(
                    self.episode
                )
            )
            return True
        else:
            return False

    def get_average_total_reward(self):
        recent_rewards = list(
            itertools.islice(
                self.total_rewards,
                max(0, len(self.total_rewards) - self.averaging_length),
                None,
            )
        )
        return sum(recent_rewards) / (
            min(len(self.total_rewards), self.averaging_length)
        )

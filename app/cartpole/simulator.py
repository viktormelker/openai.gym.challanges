from collections import deque

from app.simulator import QLearningSimulator


class CartPoleSimulator(QLearningSimulator):
    total_rewards = deque(maxlen=40)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_episode_ended(self):
        rewards = [data.reward for data in self.simulation_data]
        self.total_rewards.append(sum(rewards))
        average_total_reward = sum(self.total_rewards) / len(self.total_rewards)

        print(
            "episode: {0:4d}/{1:4d}, time_t: {2:3d}, reward: {3:8.2f}, average reward: {4:8.2f}".format(
                self.episode,
                self.num_episodes,
                self.time_t,
                self.total_rewards[-1],
                average_total_reward,
            )
        )

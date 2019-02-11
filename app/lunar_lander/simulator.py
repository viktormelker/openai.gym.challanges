from collections import deque

from app.simulator import QLearningSimulator


class LunarLanderSimulator(QLearningSimulator):
    total_rewards = deque(maxlen=40)

    def __init__(self, target_average_reward, **kwargs):
        super().__init__(**kwargs)
        self.target_average_reward = target_average_reward

    def on_training_finished(self):
        self.policy.save_weights()

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

    def is_finished(self):
        average_total_reward = sum(self.total_rewards) / len(self.total_rewards)

        if average_total_reward > self.target_average_reward:
            print(
                f"Successfully finished the challenge in {self.episode} training runs!"
            )
            return True
        else:
            return False

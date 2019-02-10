import numpy as np

from app.lunar_lander.agents import ddqnAgent as agent
from app.lunar_lander.environment import env
from collections import deque

EPISODES = 100

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]
    time_reward = 0

    total_rewards = deque(maxlen=10)
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        rewards = []
        for time_t in range(500):
            if episode > 0:
                env.render()

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            reward += time_reward

            next_state = np.reshape(next_state, [1, state_size])

            rewards.append(reward)
            agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        print(
            "episode: {0:4d}/{1:4d}, time_t: {2:3d}, reward: {3:8.2f}, average reward: {4:8.2f}".format(
                episode,
                EPISODES,
                time_t,
                total_reward,
                sum(total_rewards) / len(total_rewards),
            )
        )

    agent.save_weights()

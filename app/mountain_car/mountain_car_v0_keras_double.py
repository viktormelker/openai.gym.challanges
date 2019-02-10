import numpy as np

from app.mountain_car.agents import ddqnAgent as agent
from app.mountain_car.environment import env, success_x_pos
from collections import deque

EPISODES = 150

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]

    best_positions = deque(maxlen=20)

    for episode in range(EPISODES):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        rewards = []
        positions = [-100]
        for time_t in range(500):

            if episode > 40:
                env.render()

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            positions.append(next_state[0, 0])

            rewards.append(reward)

            agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                # print the score and break out of the loop
                best_positions.append(max(positions))
                average_best_pos = sum(best_positions) / len(best_positions)

                print(
                    "episode: {0:4d}/{1:4d}, time_t: {2:3d}, max pos: {3:6.2f}, avg pos: {4:6.2f}, total reward: {5:6.2f}".format(
                        episode,
                        EPISODES,
                        time_t,
                        max(positions),
                        average_best_pos,
                        sum(rewards),
                    )
                )
                break

        if average_best_pos > success_x_pos:
            print(f"Successfully finished the challenge in {episode} training runs!")
            break

    agent.save_weights()

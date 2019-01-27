import gym
import numpy as np

from app.policy import DoubleDQNAgent
from collections import deque

EPISODES = 1000

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('MountainCar-v0')
    # MountainCarContinuous-v0
    state_size = env.observation_space.shape[0]
    agent = DoubleDQNAgent(state_size=state_size, action_size=env.action_space.n)

    best_positions = deque(maxlen=20)
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        rewards = []
        positions = [-100]
        for time_t in range(500):
            # turn this on if you want to render
            if e > 20:
                env.render()

            # Decide action
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            if next_state[0, 0] >= 0.5:
                reward = 1000  # we did it!
            if next_state[0, 0] > max(positions):
                reward = next_state[0, 0]

            positions.append(next_state[0, 0])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            agent.target_train()

            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                best_positions.append(max(positions))

                print(
                    "episode: {0:4d}/{1:4d}, time_t: {2:3d}, max pos: {3:8.4f}, avg pos: {4:8.4f}".format(
                        e, EPISODES, time_t, max(positions),
                        sum(best_positions)/len(best_positions)
                    )
                )
                break

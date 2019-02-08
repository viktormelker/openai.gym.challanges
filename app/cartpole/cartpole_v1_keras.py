import gym
import numpy as np

from app.policy import DQNAgent
from collections import deque

EPISODES = 1000

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    agent = DQNAgent(state_size=state_size, action_size=env.action_space.n)

    survival_times = deque(maxlen=20)
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        rewards = []
        positions = []
        for time_t in range(500):
            # turn this on if you want to render
            env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            positions.append(next_state[0, 0])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                survival_times.append(time_t)
                avg_survival = sum(survival_times) / len(survival_times)
                print(
                    "episode: {}/{}, time: {} ({})".format(
                        e, EPISODES, time_t, avg_survival
                    )
                )
                break
        # train the agent with the experience of the episode
        agent.replay(32)

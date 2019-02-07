import gym
import numpy as np

from app.policy import DoubleDQNAgent
from collections import deque

EPISODES = 25

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    agent = DoubleDQNAgent(
        state_size=state_size, action_size=env.action_space.n,
        gamma=0.99, epsilon_min=0.01, weight_file='app/lunar_lander/weights/DQNN_weights.h5')
    time_reward = -0.25

    total_rewards = deque(maxlen=10)
    # Iterate the game
    for episode in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        rewards = []
        for time_t in range(500):
            # turn this on if you want to render
            if episode > 30:
                env.render()

            # Decide action
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward += time_reward

            next_state = np.reshape(next_state, [1, state_size])

            # Remember the previous state, action, reward, and done
            rewards.append(reward)
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # train agent
            agent.replay()
            agent.target_train()

            if done:
                # print the score and break out of the loop
                break

        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        print(
            "episode: {}/{}, time_t: {}, reward: {}, average reward: {}".format(
                episode, EPISODES, time_t, total_reward, sum(total_rewards)/len(total_rewards)))

    agent.save_weights()

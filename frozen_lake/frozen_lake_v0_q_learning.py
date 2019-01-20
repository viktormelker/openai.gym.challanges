# taken from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

from collections import deque

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 10000
max_steps = 100

# create lists to contain total rewards and steps per episode
reward_queue = deque()


for attempt in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    total_reward = 0
    step = 0
    # The Q-Table learning algorithm
    for i in range(max_steps):
        step += 1
        # Choose an action by greedily (with noise) picking from Q table
        action = (np.argmax(Q[s, :] +
            np.random.randn(1, env.action_space.n) * (1./(attempt + 1))))
        # Get new state and reward from environment
        observation, reward, done, _ = env.step(action)
        # Update Q-Table with new knowledge
        Q[s, action] = Q[s, action] + lr * (reward + y * np.max(Q[observation, :]) - Q[s, action])

        total_reward += reward
        s = observation
        if done is True:
            reward_queue.append(total_reward)
            break

print("Score over time: " + str(sum(reward_queue) / num_episodes))
print("Final Q-Table Values")
print(Q)

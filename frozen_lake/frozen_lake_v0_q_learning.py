# taken from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 20000

# create lists to contain total rewards and steps per episode
rList = []

for attempt in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    done = False
    step = 0
    # The Q-Table learning algorithm
    while step < 99:
        step += 1
        # Choose an action by greedily (with noise) picking from Q table
        action = (np.argmax(Q[s, :] +
            np.random.randn(1, env.action_space.n) * (1./(attempt + 1))))
        # Get new state and reward from environment
        observation, reward, done, _ = env.step(action)
        # Update Q-Table with new knowledge
        Q[s, action] = Q[s, action] + lr * (reward + y * np.max(Q[observation, :]) - Q[s, action])

        rAll += reward
        s = observation
        if done is True:
            break
    rList.append(rAll)

print "Score over time: " + str(sum(rList) / num_episodes)
print "Final Q-Table Values"
print Q

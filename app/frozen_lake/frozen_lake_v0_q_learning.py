# taken from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

from collections import deque

import gym

import numpy as np

from app.policies.q_learning import QTablePolicy

env = gym.make("FrozenLake-v0")

num_episodes = 10000
max_steps = 100
reward_queue = deque()

policy = QTablePolicy(
    state_size=env.observation_space.n, action_size=env.action_space.n
)

for attempt in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 1])
    states = []
    actions = []
    for i in range(max_steps):
        states.append(state)

        action = policy.get_action(state, attempt)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 1])
        policy.update(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        state = next_state

        if done is True:
            break

print("Score over time: " + str(sum(reward_queue) / num_episodes))
print("Final Q-Table Values")
print(policy.Q)

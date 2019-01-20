# taken from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

from collections import deque

import gym

from policy import QTablePolicy

env = gym.make('FrozenLake-v0')

num_episodes = 10000
max_steps = 100
reward_queue = deque()
time_reward = 0

policy = QTablePolicy(num_actions=env.action_space.n,
                      num_states=env.observation_space.n)

for attempt in range(num_episodes):
    total_reward = 0
    state = env.reset()
    states = []
    actions = []
    for i in range(max_steps):
        states.append(state)

        action = policy.get_action(state, attempt)
        actions.append(action)

        state, reward, done, _ = env.step(action)
        reward += time_reward

        total_reward += reward
        policy.update(states, actions, total_reward, result_state=state)

        if done is True:
            reward_queue.append(total_reward)
            break

print("Score over time: " + str(sum(reward_queue) / num_episodes))
print("Final Q-Table Values")
print(policy.Q)

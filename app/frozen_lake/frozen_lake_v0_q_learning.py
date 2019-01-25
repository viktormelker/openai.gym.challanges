# taken from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

from collections import deque

import gym

from app.policy import QTablePolicy

env = gym.make('FrozenLake-v0')

num_episodes = 10000
max_steps = 100
reward_queue = deque()

policy = QTablePolicy(num_actions=env.action_space.n,
                      num_states=env.observation_space.n)

for attempt in range(num_episodes):
    state = env.reset()
    states = []
    actions = []
    for i in range(max_steps):
        states.append(state)

        action = policy.get_action(state, attempt)
        actions.append(action)

        state, reward, done, _ = env.step(action)
        policy.update(states, actions, reward, result_state=state)

        if done is True:
            break

print("Score over time: " + str(sum(reward_queue) / num_episodes))
print("Final Q-Table Values")
print(policy.Q)

from collections import deque

import gym

from app.policy import SimplePolicy

env = gym.make("FrozenLake-v0")

num_episodes = 10000
max_steps = 100
reward_queue = deque()
time_reward = 0

policy = SimplePolicy(
    num_actions=env.action_space.n, num_states=env.observation_space.n
)

for attempt in range(num_episodes):
    total_reward = 0
    state = env.reset()
    states = []
    actions = []
    for i in range(max_steps):
        states.append(state)

        action = policy.get_action(state)
        actions.append(action)

        state, reward, done, _ = env.step(action)
        reward += time_reward

        total_reward += reward
        policy.update(states, actions, total_reward)

        if done is True:
            reward_queue.append(total_reward)
            break

print("Score over time: " + str(sum(reward_queue) / num_episodes))
print(policy.action_probabilities)

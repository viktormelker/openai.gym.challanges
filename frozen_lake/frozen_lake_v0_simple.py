import gym
from collections import deque
from policy import Policy

env = gym.make('FrozenLake-v0')


num_episodes = 10000
max_steps = 100
discount_factor = 1
reward_queue = deque()
time_reward = 0

policy = Policy(num_actions=env.action_space.n,
                num_states=env.observation_space.n)

for attempt in range(num_episodes):
    total_reward = 0
    observation = env.reset()
    states = []
    actions = []
    for i in range(max_steps):
        #env.render()

        states.append(observation)

        action = policy.get_action(observation)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        reward += time_reward

        total_reward = total_reward * discount_factor + reward
        policy.update(states, actions, total_reward)

        if done is True:
            reward_queue.append(total_reward)
            break

print(policy.action_probabilities)
print("Score over time: " + str(sum(reward_queue) / num_episodes))

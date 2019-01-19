import gym
import numpy as np
from collections import deque

env = gym.make('FrozenLake-v0')


class Policy:
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    learning_rate = 0.1
    age_factor = 0.9

    action_probabilities = np.array(16 * [4 * [0.25]])

    def update(self, states, actions, reward):
        age = len(states) - 1
        for state, action in zip(states, actions):
            self.action_probabilities[state, action] += max(0, (
                self.learning_rate * reward * pow(self.age_factor, age)
            ))

            self.action_probabilities[state] = (
                self.action_probabilities[state] / self.action_probabilities[state].sum())
            age -= 1

    def get_action(self, state):
        return np.random.choice(
            self.num_actions, 1, p=self.action_probabilities[state]
        )[0]


num_episodes = 10000
max_steps = 100
discount_factor = 1
reward_queue = deque(maxlen=50)
time_reward = 0

policy = Policy()

for episode in range(num_episodes):
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

        if done is True:
            policy.update(states, actions, total_reward)
            reward_queue.append(total_reward)
            print(f'Game finished with total reward {total_reward}. Rolling average reward: {sum(reward_queue)/len(reward_queue)}')
            break

print(policy.action_probabilities)

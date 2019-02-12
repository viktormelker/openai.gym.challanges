from app.cartpole.environment import env
from app.policies.q_learning import DoubleDQNAgent, DQNAgent, RandomPolicy

state_size = env.observation_space.shape[0]

# Double Deep Q learning
ddqnAgent = DoubleDQNAgent(state_size=state_size, action_size=env.action_space.n)

# Deep Q learning
dqnAgent = DQNAgent(state_size=state_size, action_size=env.action_space.n)

random_policy = RandomPolicy(state_size=state_size, action_size=env.action_space.n)

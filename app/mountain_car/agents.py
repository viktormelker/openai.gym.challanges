
from app.mountain_car.environment import env
from app.mountain_car.reward_functions import potential_function
from app.policy import DoubleDQNAgent, DQNAgent


state_size = env.observation_space.shape[0]

# Double Deep Q learning
ddqnAgent = DoubleDQNAgent(
    state_size=state_size, action_size=env.action_space.n)

# Deep Q learning
dqnAgent = DQNAgent(state_size=state_size, action_size=env.action_space.n)

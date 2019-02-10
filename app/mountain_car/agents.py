from app.mountain_car.environment import env
from app.mountain_car.reward_functions import potential_function
from app.policy import DoubleDQNAgent, DQNAgent, RandomPolicy


state_size = env.observation_space.shape[0]

# Double Deep Q learning
ddqnAgent = DoubleDQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    potential_function=potential_function,
    learning_rate=0.003,
    weight_file="app/mountain_car/weights/DDQNN_weights_1.h5",
)

# Deep Q learning
dqnAgent = DQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    potential_function=potential_function,
    weight_file="app/mountain_car/weights/DQNN_weights_1.h5",
)

random_policy = RandomPolicy(
    state_size=state_size,
    action_size=env.action_space.n,
)

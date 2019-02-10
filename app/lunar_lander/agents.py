from app.lunar_lander.environment import env
from app.policies.q_learning import DoubleDQNAgent, DQNAgent, RandomPolicy


state_size = env.observation_space.shape[0]

# Double Deep Q learning
ddqnAgent = DoubleDQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    learning_rate=0.001,
    weight_file="app/lunar_lander/weights/DQNN_weights_2.h5",
)

# Deep Q learning
dqnAgent = DQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    weight_file="app/lunar_lander/weights/DQN_weights_1.h5",
)

random_policy = RandomPolicy(
    state_size=state_size, action_size=env.action_space.n)

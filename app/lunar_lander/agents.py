from app.lunar_lander.environment import env
from app.lunar_lander.reward_functions import potential_function
from app.policies.q_learning import DoubleDQNAgent, DQNAgent, RandomPolicy

state_size = env.observation_space.shape[0]

# Double Deep Q learning
ddqnAgent = DoubleDQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    weight_file="app/lunar_lander/weights/DQNN_weights_3.h5",
    potential_function=potential_function,
)

# Deep Q learning
dqnAgent = DQNAgent(
    state_size=state_size,
    action_size=env.action_space.n,
    weight_file="app/lunar_lander/weights/DQN_weights_2.h5",
    gamma=0.99,
    potential_function=potential_function,
    learning_rate=0.02,
)

random_policy = RandomPolicy(state_size=state_size, action_size=env.action_space.n)


def get_current_agent(**kwargs):
    params = {
        "state_size": state_size,
        "action_size": env.action_space.n,
        "weight_file": "app/lunar_lander/weights/DQN_weights_2.h5",
        "gamma": 0.99,
        "potential_function": potential_function,
        "learning_rate": 0.02,
    }
    params.update(**kwargs)
    agent = DQNAgent(**params)
    return agent

from app.mountain_car.agents import dqnAgent as policy
from app.mountain_car.environment import env
from app.mountain_car.simulator import MountainCarSimulator

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]

    simulator = MountainCarSimulator(
        policy=policy,
        environment=env,
        state_size=state_size,
        render_gui=True,
        num_episodes=50,
        success_x_pos=0.3,
    )

    simulator.simulate()

#!/usr/bin/env python3

from mountain_car.agents import dqnAgent as policy
from mountain_car.environment import env
from mountain_car.simulator import MountainCarSimulator

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

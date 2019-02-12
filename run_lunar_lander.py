#!/usr/bin/env python3

from app.lunar_lander.agents import dqnAgent as policy
from app.lunar_lander.environment import env
from app.lunar_lander.simulator import LunarLanderSimulator

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]

    simulator = LunarLanderSimulator(
        policy=policy,
        environment=env,
        state_size=state_size,
        render_gui=True,
        num_episodes=100,
        target_average_reward=100,
    )

    simulator.simulate()

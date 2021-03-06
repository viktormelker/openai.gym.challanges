#!/usr/bin/env python3

from cartpole.agents import dqnAgent as policy
from cartpole.environment import env
from cartpole.simulator import CartPoleSimulator

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]

    simulator = CartPoleSimulator(
        policy=policy,
        environment=env,
        state_size=state_size,
        render_gui=True,
        num_episodes=1000,
    )

    simulator.simulate()

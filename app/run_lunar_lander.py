#!/usr/bin/env python3

import argparse

from app.lunar_lander.agents import get_current_agent
from app.lunar_lander.environment import env
from app.lunar_lander.simulator import LunarLanderSimulator


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", help="discount rate for", type=float, default=0.95)
    parser.add_argument(
        "--learning_rate", help="learning rate for", type=float, default=0.001
    )
    parser.add_argument(
        "--job-dir",
        help="Output directory",
        type=str,
        default="app/lunar_lander/weights",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    state_size = env.observation_space.shape[0]

    policy = get_current_agent(**args.__dict__)

    simulator = LunarLanderSimulator(
        policy=policy,
        environment=env,
        state_size=state_size,
        render_gui=False,
        num_episodes=3,
        target_average_reward=100,
    )

    simulator.simulate()

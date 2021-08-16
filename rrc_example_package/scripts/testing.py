#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment
This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import sys

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy


def main():

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        rearrange_dice_env.ActionType.POSITION,
        goal=None,
        step_size=1,
    )
    camera_params = env.camera_params
    goal = env.goal
    print(camera_params)
    print(goal)

    #observation = env.reset()


if __name__ == "__main__":
    main()

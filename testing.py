#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment
This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
import trifinger_simulation.tasks.rearrange_dice as task


def main():

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        rearrange_dice_env.ActionType.POSITION,
        goal= None,
        step_size=1,
    )
    env.reset()

    camera_observation = env.platform.get_camera_observation(0)

    for c in camera_observation.cameras:
        print(c.image.shape)
    segmentation_masks = [
            segment_image(c.image) for c in camera_observation.cameras
        ]

    camera_params = env.camera_params
    goal = env.goal
    #masks = task.generate_goal_mask(camera_params, goal)
    #np.save('masks.npy', masks)

if __name__ == "__main__":
    main()
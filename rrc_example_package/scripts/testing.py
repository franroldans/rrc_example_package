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

def _get_cell_corners_3d(
    pos: Position,
) -> np.ndarray:
    """Get 3d positions of the corners of the cell at the given position."""
    d = DIE_WIDTH / 2
    nppos = np.asarray(pos)

    # order of the corners is the same as in the cube model of the
    # trifinger_object_tracking package
    # people.tue.mpg.de/mpi-is-software/robotfingers/docs/trifinger_object_tracking/doc/cube_model.html
    return np.array(
        (
            nppos + (d, -d, d),
            nppos + (d, d, d),
            nppos + (-d, d, d),
            nppos + (-d, -d, d),
            nppos + (d, -d, -d),
            nppos + (d, d, -d),
            nppos + (-d, d, -d),
            nppos + (-d, -d, -d),
        )
    )

def calculate_XYZ(pos, tvec, rmat, camera_matrix):
                                      
    #Solve: From Image Pixels, find World Points
    scalingfactor = 1 / camera_matrix[2, 2]
    uv_1=np.array([[pos[0],pos[1],1]], dtype=np.float32)
    uv_1=uv_1.T
    suv_1=scalingfactor*uv_1
    inverse_cam_mtx =  np.linalg.inv(camera_matrix)
    inverse_r_mtx = np.linalg.inv(rmat)
    xyz_c=inverse_cam_mtx.dot(suv_1)
    xyz_c=xyz_c-tvec
    XYZ=inverse_r_mtx.dot(xyz_c)
    return XYZ

def main():

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        rearrange_dice_env.ActionType.POSITION,
        goal= task.sample_goal(),
        step_size=1,
    )
    camera_params = env.camera_params
    goal = env.goal
    print(camera_params)
    print(goal)
    masks = task.generate_goal_mask([camera_params], goal)
    print(masks)

    # get camera position and orientation separately
    tvec = camera_params.tf_world_to_camera[:3, 3]
    rmat = camera_params.tf_world_to_camera[:3, :3]
    rvec = Rotation.from_matrix(rmat).as_rotvec()


    for pos in goal:
        #corners = _get_cell_corners_3d(pos)
        img_plane = []

        # project corner points into the image
        proj_pos, _ = cv2.projectPoints(
            pos,
            rvec,
            tvec,
            camera_params.camera_matrix,
            camera_params.distortion_coefficients,
        )
        img_plane.append(proj_pos)

    for pos in img_plane:
        xyz = calculate_XYZ(pos, tvec, rmat, camera_params.camera_matrix)




    #observation = env.reset()


if __name__ == "__main__":
    main()
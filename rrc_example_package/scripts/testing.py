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


def calculate_XYZ(pos, tvec, rmat, camera_matrix):
                                      
    #Solve: From Image Pixels, find World Points
    scalingfactor = 1 / camera_matrix[2, 2]
    uv_1=np.array([[pos[0][0][0],pos[0][0][1],1]], dtype=np.float32)
    uv_1=uv_1.T
    suv_1=scalingfactor*uv_1
    inverse_cam_mtx =  np.linalg.inv(camera_matrix)
    inverse_r_mtx = np.linalg.inv(rmat)
    xyz_c=inverse_cam_mtx.dot(suv_1)
    xyz_c=xyz_c-tvec
    print(tvec.shape)
    print(xyz_c.shape)
    print(inverse_r_mtx.shape)
    XYZ=inverse_r_mtx.dot(xyz_c)
    print(XYZ.shape)
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
    masks = task.generate_goal_mask(camera_params, goal)
    print(masks)
    np.save('masks.npy', masks)

    # get camera position and orientation separately
    tvec = camera_params[0].tf_world_to_camera[:3, 3]
    print(tvec)
    rmat = camera_params[0].tf_world_to_camera[:3, :3]
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    img_plane = []
    for pos in goal:
        #corners = _get_cell_corners_3d(pos)

        # project corner points into the image
        proj_pos, _ = cv2.projectPoints(
            pos,
            rvec,
            tvec,
            camera_params[0].camera_matrix,
            camera_params[0].distortion_coefficients,
        )
        img_plane.append(proj_pos)
    #print(img_plane[0][0][0])

    for i, pos in enumerate(img_plane):
        print(pos)
        xyz = calculate_XYZ(pos, tvec, rmat, camera_params[0].camera_matrix)
        print(goal[i])
        print(xyz)




    #observation = env.reset()


if __name__ == "__main__":
    main()
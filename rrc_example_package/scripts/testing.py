#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment
This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import sys
import numpy as np
import cv2
import imutils

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image


def image2world(image_point, camera_parameters, z = 0.011):
    
    # get camera position and orientation separately
    tvec = camera_parameters.tf_world_to_camera[:3, 3]
    tvec = tvec[:, np.newaxis]
    rmat = camera_parameters.tf_world_to_camera[:3, :3]
    camMat = np.asarray(camera_parameters.camera_matrix)
    iRot = np.linalg.inv(rmat)
    iCam = np.linalg.inv(camMat)

    uvPoint = np.ones((3, 1))
    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
    wcPoint[2] = z #Hardcoded as z is always 0.011 if constrained to only push cube
    return tuple(map(float,wcPoint))

def get_2d_center(x, y, w, h):
    return (round((x + x + w) / 2), round((y+y+h) / 2))
    

def image2coords(camera_observation, camera_params, write_images=False):
    len_out = 0
    for i, c in enumerate(camera_observation.cameras):
        copy = c.image.copy()
        grey = cv2.cvtColor(c.image, cv2.COLOR_BGR2GRAY)
        grey = grey * segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
        if write_images:
            cv2.imwrite('grey{}.png'.format(i), grey)
            cv2.imwrite('seg{}.png'.format(i),segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)))
        decrease_noise = cv2.fastNlMeansDenoising(grey, 10, 15, 7, 21)
        blurred = cv2.GaussianBlur(decrease_noise, (3, 3), 0)
        canny = cv2.Canny(blurred, 10, 30)
        thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        out = []
        for c in contours:
            # obtain the bounding rectangle coordinates for each square
            x, y, w, h = cv2.boundingRect(c)
            x_c, y_c = get_2d_center(x, y, w, h)
            world_point_c = image2world((x_c, y_c), camera_params[i], z = 0.011)
            out.append([(x, y, w, h), world_point_c]) # return bboxes and 3d point
            # With the bounding rectangle coordinates, draw a green bounding boxes and its centers for visualization purposes
            cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.circle(copy, (x_c, y_c), radius=0, color=(36, 255, 12), thickness=2)
        id = i + 10
        if write_images: 
            cv2.imwrite('test{}.png'.format(id), copy)
        #temporarilly keep the view with the highest number of detections
        if len_out < len(out):
            coords = out
            len_out = len(out)
    return coords
    
def main():

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        rearrange_dice_env.ActionType.POSITION,
        goal= None,
        step_size=1,
    )
    env.reset()

    camera_observation = env.platform.get_camera_observation(0)
    camera_params = env.camera_params
    coords = image2coords(camera_observation, camera_params, True)
    print(coords)
    


if __name__ == "__main__":
    main()

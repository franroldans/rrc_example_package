#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment
This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import imutils

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image


def main():

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        rearrange_dice_env.ActionType.POSITION,
        goal= None,
        step_size=1,
    )
    env.reset()

    camera_observation = env.platform.get_camera_observation(0)

    for i, c in enumerate(camera_observation.cameras):
        cv2.imwrite('test{}.png'.format(i), c.image)
        mult = c.image * segment_image(c.image)[:,:, np.newaxis]
        cv2.imwrite('mult{}.png'.format(i), mult)
        copy = mult.copy()
        #mult = copy * segment_image(c.image)
        grey = cv2.cvtColor(c.image, cv2.COLOR_BGR2GRAY)
        decrease_noise = cv2.fastNlMeansDenoising(grey, 10, 15, 7, 21)
        blurred = cv2.GaussianBlur(decrease_noise, (3, 3), 0)
        canny = cv2.Canny(blurred, 20, 40)
        thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]

        for c in contours:
            # obtain the bounding rectangle coordinates for each square
            x, y, w, h = cv2.boundingRect(c)
            # With the bounding rectangle coordinates we draw the green bounding boxes
            cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.imwrite('test{}.png'.format(id), copy)

    """segmentation_masks = [
            segment_image(c.image) for c in camera_observation.cameras
        ]

    #segmentation_masks = np.load('masks.npy')
    for idx, mask in enumerate(segmentation_masks):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])


        result = mask.copy()
        for c in cnts:
            # get rotated rectangle from contour
            rot_rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rot_rect)
            box = np.int0(box)
            # draw rotated rectangle on copy of img
            cv2.drawContours(result,[box],0,(255,0,0),2)
        id = idx + 10
        cv2.imwrite('test{}.png'.format(id), result)"""

    #masks = task.generate_goal_mask(camera_params, goal)
    #np.save('masks.npy', masks)

if __name__ == "__main__":
    main()

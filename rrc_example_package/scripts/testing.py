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


def get_2d_center(x, y, w, h):
    return (round((x + x + w) / 2), round((y+y+h) / 2))
    

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
        copy = c.image.copy()
        grey = cv2.cvtColor(c.image, cv2.COLOR_BGR2GRAY)
        grey = grey * segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite('grey{}.png'.format(i), grey)
        cv2.imwrite('seg{}.png'.format(i),segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)))
        decrease_noise = cv2.fastNlMeansDenoising(grey, 10, 15, 7, 21)
        blurred = cv2.GaussianBlur(decrease_noise, (3, 3), 0)
        canny = cv2.Canny(blurred, 10, 30)
        thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for c in contours:
            # obtain the bounding rectangle coordinates for each square
            x, y, w, h = cv2.boundingRect(c)
            x_c, y_c = get_2d_center(x, y, w, h)
            print(x_c)
            print(y_c)
            # With the bounding rectangle coordinates we draw the green bounding boxes
            cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.circle(copy, (x_c, y_c), radius=0, color=(36, 255, 12), thickness=2)
        id = i + 10
        cv2.imwrite('test{}.png'.format(id), copy)


if __name__ == "__main__":
    main()

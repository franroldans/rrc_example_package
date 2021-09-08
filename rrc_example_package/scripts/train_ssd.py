import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import imutils
import itertools

import torchvision.models as models
import torch
from torchvision import transforms

from rrc_example_package import rearrange_dice_env
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image


FACE_CORNERS = (
    (0, 1, 2, 3),
    (4, 5, 1, 0),
    (5, 6, 2, 1),
    (7, 6, 2, 3),
    (4, 7, 3, 0),
    (4, 5, 6, 7),
)

def bbox_generator(camera_params, goal):
  mask = generate_goal_mask(camera_params, goal)
  contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = contours[0] if len(contours) == 2 else contours[1]
  x, y, w, h = cv2.boundingRect(contours)
  return (x, x + w, y, y + h)

def generate_batch(env, batch_size):
  batch = np.ones((batch_size, 3, 270, 270))
  bboxes = np.ones((batch_size, 25, 4))
  for i in range(batch_size):
    g_ = task.sample_goal()
    print(g_)
    g_mask = generate_goal_mask(env.camera_params[0], g)
    batch[i] = np.stack((g_mask,)*3, axis=-1)
    for idx, g in enumerate(g_):
      bboxes[i, idx, :] = bbox_generator(env.camera_params[0], g)
  return g_mask, bboxes

"""def generate_batch(env, batch_size):
  batch = np.ones((batch_size, 3, 270, 270))
	goals = np.ones((batch_size, 25 * 3))
	#goals = np.ones((batch_size, 25 * 2))
	for i in range(batch_size):
		seg_mask = np.ones((3, 270, 270))
		g = task.sample_goal()
		goal = list(itertools.chain(*g))
		#goal = [g for i, g in enumerate(goal) if ((i+1) % 3) !=0]
		goals[i] = np.array(goal)
		for idx, c in enumerate(env.camera_params):
			seg_mask[idx,:,:] = generate_goal_mask(c, g)
		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
		batch[i] = seg_mask
	return batch, goals"""


model = models.detection.ssd300_vgg16()
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
mask, bboxes = generate_batch(env, 1)
print(mask.shape)
print(bboxes)

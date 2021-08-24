import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import imutils

import torchvision.models as models
import torch

from rrc_example_package import rearrange_dice_env
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image



class ResNet(torch.nn.Module):


	def __init__(self, resnet):
		super(ResNet, self).__init__()
		self.resnet = resnet
		self.fc = torch.nn.Linear(1000, 3*25)

	def forward(self, x):
		x = self.resnet(x)
		x = torch.flatten(x, 1)
		out = self.fc(x)
		return x
	





"""def generate_batch(batch_size):
	batch = np.ones((batch_size, 270, 270, 3))
	for i in range(batch_size):
		seg_mask = np.ones((270, 270, 3))
		env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
		env.reset()
		obs = env.platform.get_camera_observation(0)
		for idx, c in enumerate(obs.cameras):
			seg_mask[:, :, idx]  = segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) 
		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
		batch[i] = seg_mask
	return batch"""

def generate_batch(env, batch_size):
	batch = np.ones((batch_size, 270, 270, 3))
	goals = np.ones((batch_size, 25 * 3))
	for i in range(batch_size):
		seg_mask = np.ones((270, 270, 3))
		g = task.sample_goal()
		goal = np.array(list(g))
		goals[i] = goal
		for idx, c in enumerate(env.camera_params):
			seg_mask[:,:,idx] = task.generate_goal_mask(c, g)
		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
		batch[i] = seg_mask
	return batch


resnet = models.resnet18(pretrained=False)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet = ResNet(newmodel)
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
batch = generate_batch(env, 64)
#loss = nn.MSELoss()

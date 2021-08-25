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

class ResNet(torch.nn.Module):


	def __init__(self, resnet):
		super(ResNet, self).__init__()
		self.resnet = resnet
		self.fc = torch.nn.Linear(512, 3*25)

	def forward(self, x):
		x = self.resnet(x)
		print(x.shape)
		x = torch.flatten(x, 1)
		print(x.shape)
		out = self.fc(x)
		return out
	

def generate_batch(env, batch_size):
	batch = np.ones((batch_size, 3, 270, 270))
	goals = np.ones((batch_size, 25 * 3))
	for i in range(batch_size):
		seg_mask = np.ones((3, 270, 170))
		g = task.sample_goal()
		goal = np.array(list(itertools.chain(*g)))
		goals[i] = goal
		for idx, c in enumerate(env.camera_params):
			seg_mask[idx,:,:] = generate_goal_mask(c, g)
		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
		batch[i] = seg_mask
	return batch, goals

def get_cell_corners_3d(pos):
    """Get 3d positions of the corners of the cell at the given position."""
    d = 0.022 / 2
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


def generate_goal_mask(camera_parameters, goal):
    """Generate goal masks that can be used with :func:`evaluate_state`.

    A goal mask is a single-channel image where the areas at which dice are
    supposed to be placed are white and everything else is black.  So it
    corresponds more or less to a segmentation mask where all dice are at the
    goal positions.

    For rendering the mask, :data:`TARGET_WIDTH` is used for the die width to
    add some tolerance.

    Args:
        camera_parameters: List of camera parameters, one per camera.
        goal: The goal die positions.

    Returns:
        List of masks.  The number and order of masks corresponds to the input
        ``camera_parameters``.
    """
    #masks = []
    #for cam in camera_parameters:
    #mask = np.zeros((camera_parameters.image_height, camera_parameters.image_width), dtype=np.uint8)
    mask = np.zeros((270, 270), dtype=np.uint8)

    # get camera position and orientation separately
    tvec = camera_parameters.tf_world_to_camera[:3, 3]
    rmat = camera_parameters.tf_world_to_camera[:3, :3]
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    for pos in goal:
        corners = get_cell_corners_3d(pos)

        # project corner points into the image
        projected_corners, _ = cv2.projectPoints(
            corners,
            rvec,
            tvec,
            camera_parameters.camera_matrix,
            camera_parameters.distortion_coefficients,
        )

        # draw faces in mask
        for face_corner_idx in FACE_CORNERS:
            points = np.array(
                [projected_corners[i] for i in face_corner_idx],
                dtype=np.int32,
            )
            mask = cv2.fillConvexPoly(mask, points, 255)

        #masks.append(mask)

    return mask

resnet_ = models.resnet18(pretrained=False)
newmodel = torch.nn.Sequential(*(list(resnet_.children())[:-1]))
resnet = ResNet(newmodel)

print(resnet)
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
while True:

    input_batch, goals = generate_batch(env, 8)
    input_batch = torch.from_numpy(input_batch).float()
    goals = torch.from_numpy(goals).float()
    loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        goals = goals.to('cuda')
        resnet.to('cuda')

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    #input_batch = preprocess(input_batch)
    out = resnet(input_batch)
    print(out.shape)
    print(goals.shape)
    cost = loss(out, goals)
    cost.backward()
    print("Loss: {}".format(cost))


#loss = nn.MSELoss()

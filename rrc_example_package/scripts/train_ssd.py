import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import imutils
import itertools

import torchvision.models as models
import torch
from torchvision import transforms
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from rrc_example_package import rearrange_dice_env
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

def my_ssd300_vgg16(pretrained= False, progress= True, num_classes= 91,
                 pretrained_backbone = True, trainable_backbone_layers = None, **kwargs):
    """Constructs an SSD model with input size 300x300 and a VGG16 backbone.

    Reference: `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        >>> model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = models.detection.ssd._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = models.detection.ssd._vgg_extractor("vgg16_features", False, progress, pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                           steps=[8, 16, 32, 64, 100, 300])

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = models.detection.ssd.SSD(backbone, anchor_generator, (270, 270), num_classes, **kwargs)
    return model

FACE_CORNERS = (
    (0, 1, 2, 3),
    (4, 5, 1, 0),
    (5, 6, 2, 1),
    (7, 6, 2, 3),
    (4, 7, 3, 0),
    (4, 5, 6, 7),
)

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


def bbox_generator(camera_params, goal, i):
  #print(goal)
  mask = generate_goal_mask(camera_params, [goal])
  #cv2.imwrite('mask{}.png'.format(i), mask)
  contour = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
  #contour = contour[0] if len(contour) == 2 else contour[1]
  x, y, w, h = cv2.boundingRect(contour[0])
  return np.array([x, y, x + w, y + h])

def generate_batch(env, batch_size):
  print(env.camera_params[0])
  batch = np.ones((batch_size, 3, 270, 270))
  detections = []
  for i in range(batch_size):
    g_ = task.sample_goal()
    #print(g_)
    g_mask = generate_goal_mask(env.camera_params[0], g_)
    batch[i] = np.stack((g_mask,)*3, axis=0)
    bboxes=[]
    for idx, g in enumerate(g_):
        bboxes.append(bbox_generator(env.camera_params[0], g, idx))
    detections.append({'boxes': torch.from_numpy(np.array(bboxes)).float(), 
		   'labels': np.array([1])})
    #detections.append(bboxes)
  return batch, detections

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


model = my_ssd300_vgg16(num_classes=1)
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
#mask, bboxes = generate_batch(env, 16)
#print(bboxes)


while True:
  mask, bboxes = generate_batch(env, 16)
  model.forward(torch.from_numpy(mask).float(), bboxes)

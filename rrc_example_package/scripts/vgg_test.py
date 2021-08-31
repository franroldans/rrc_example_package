import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import imutils
import itertools

from torchvision import transforms

from rrc_example_package import rearrange_dice_env
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


from typing import Union, List, Dict, Any, cast


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2*25),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


FACE_CORNERS = (
    (0, 1, 2, 3),
    (4, 5, 1, 0),
    (5, 6, 2, 1),
    (7, 6, 2, 3),
    (4, 7, 3, 0),
    (4, 5, 6, 7),
)

class CustomResNet(torch.nn.Module):


	def __init__(self, resnet):
		super(CustomResNet, self).__init__()
		self.resnet = resnet
		self.fc = torch.nn.Linear(512, 2*25)

	def forward(self, x):
		x = self.resnet(x)
		x = torch.flatten(x, 1)
		out = self.fc(x)
		#out = torch.clamp(out, -0.15, 0.15)
		return out
	

def generate_batch(env, batch_size):
	batch = np.ones((batch_size,1, 270, 270))
	goals = np.ones((batch_size, 25 * 2))
	#goals = np.ones((batch_size, 25 * 2))
	for i in range(batch_size):
		#seg_mask = np.ones((3, 270, 270))
		g_ = task.sample_goal()
		g = world2image(g_, env.camera_params[0])
		goal = list(itertools.chain(*g))
		#goal = [g for i, g in enumerate(goal) if ((i+1) % 3) !=0]
		goals[i] = np.array(goal)
		#for idx, c in enumerate(env.camera_params):
		batch[i] = np.expand_dims(generate_goal_mask(env.camera_params[0], g_), axis=0)
		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
		#batch[i] = seg_mask
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

def world2image(goal, camera_params):
    img_plane = []
    # get camera position and orientation separately
    tvec = camera_params.tf_world_to_camera[:3, 3]
    rmat = camera_params.tf_world_to_camera[:3, :3]
    rvec = Rotation.from_matrix(rmat).as_rotvec()
    for pos in goal:
        # project corner points into the image
        proj_pos, _ = cv2.projectPoints(
            pos,
            rvec,
            tvec,
            camera_params.camera_matrix,
            camera_params.distortion_coefficients,
        )
        img_plane.append(proj_pos[0][0])
    return img_plane

resnet_ = resnet18(pretrained=False)
newmodel = torch.nn.Sequential(*(list(resnet_.children())[:-1]))
resnet = CustomResNet(newmodel)

print(resnet.parameters())
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.0001)
min_cost = 300000
while True:

    input_batch, goals = generate_batch(env, 16)
    input_batch = torch.from_numpy(input_batch).float()
    goals = torch.from_numpy(goals).float()
    loss = torch.nn.MSELoss()
    #loss = torch.norm()
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        goals = goals.to('cuda')
        resnet.to('cuda')

    """preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])"""

    #input_batch = preprocess(input_batch)
    input_batch = torch.nn.functional.normalize(input_batch)
    out = resnet(input_batch)
    """cost = 0
    for i in range(0, 25*3, 3):
    	cost += torch.norm(out[i:i+3] - goals[i:i+3], 2)
    cost = cost / 25"""
    cost = loss(out, goals)
    cost.backward()
    optim.step()
    print("Loss: {}".format(cost))
    if cost < min_cost:
        min_cost = cost
        torch.save(resnet.state_dict(), './best_model_resnet_2D.pth')


#loss = nn.MSELoss()

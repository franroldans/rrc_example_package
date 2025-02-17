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
import torch.nn.functional as F
import warnings
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import vgg
from torchvision.ops import boxes as box_ops
from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['SSD', 'ssd300_vgg16']

model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
}

backbone_urls = {
    # We port the features of a VGG16 backbone trained by amdegroot because unlike the one on TorchVision, it uses the
    # same input standardization method as the paper. Ref: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
    'vgg16_features': 'https://download.pytorch.org/models/vgg16_features-amdegroot.pth'
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)



class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'bbox_regression': self.regression_head(x),
            'cls_logits': self.classification_head(x),
        }


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.module_list:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)

		
class SSD(nn.Module):
    """
    Implements SSD architecture from `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

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

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute with the list of the output channels of
            each feature map. The backbone should return a single Tensor or an OrderedDict[Tensor].
        anchor_generator (DefaultBoxGenerator): module that generates the default boxes for a
            set of feature maps.
        size (Tuple[int, int]): the width and height to which images will be rescaled before feeding them
            to the backbone.
        num_classes (int): number of output classes of the model (excluding the background).
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        head (nn.Module, optional): Module run on top of the backbone features. Defaults to a module containing
            a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        topk_candidates (int): Number of best detections to keep before NMS.
        positive_fraction (float): a number between 0 and 1 which indicates the proportion of positive
            proposals used during the training of the classification head. It is used to estimate the negative to
            positive ratio.
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator,
                 size: Tuple[int, int], num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 head: Optional[nn.Module] = None,
                 score_thresh: float = 0.00,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 25,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25):
        super().__init__()

        self.backbone = backbone

        self.anchor_generator = anchor_generator

        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))

        if head is None:
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = det_utils.retrieve_out_channels(backbone, size)

            assert len(out_channels) == len(anchor_generator.aspect_ratios)

            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head

        self.proposal_matcher = det_utils.SSDMatcher(iou_thresh)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min(size), max(size), image_mean, image_std,
                                                  size_divisible=1, fixed_size=size)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

        # used only on torchscript mode
        self._has_warned = False

    def eager_outputs(self, losses: Dict[str, Tensor],
                      detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs['bbox_regression']
        cls_logits = head_outputs['cls_logits']

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image,
             matched_idxs_per_image) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ))

           
        bbox_loss = torch.stack(bbox_loss)
        N = max(1, num_foreground)
        out_loss = bbox_loss.sum() / N
        print("Loss: {}".format(out_loss))
        return out_loss


    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image['boxes'].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                                   device=anchors_per_image.device))
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            print(scores)
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            print(boxes.shape)
            print(boxes)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(0, num_classes):
                score = np.stack(([1],)*len(boxes))

                #keep_idxs = score > self.score_thresh
                #score = score[keep_idxs]
                #box = boxes[keep_idxs]

                # keep only topk scoring predictions
                #num_topk = min(self.topk_candidates, 25)
                #score, idxs = score.topk(num_topk)
                #box = boxes[idxs]

                image_boxes.append(boxes)
                image_scores.append(score)
                image_labels.append(torch.full_like(torch.from_numpy(score), fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.from_numpy(np.array(image_scores))
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections


class SSDFeatureExtractorVGG(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(
            *backbone[:maxpool4_pos]  # until conv4_3
        )

        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.ReLU(inplace=True),
            )
        ])
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            fc,
        ))
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # L2 regularization + Rescaling of 1st block's feature map
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _vgg_extractor(backbone_name: str, highres: bool, progress: bool, pretrained: bool, trainable_layers: int):
    if backbone_name in backbone_urls:
        # Use custom backbones more appropriate for SSD
        arch = backbone_name.split('_')[0]
        backbone = vgg.__dict__[arch](pretrained=False, progress=progress).features
        if pretrained:
            state_dict = load_state_dict_from_url(backbone_urls[backbone_name], progress=progress)
            backbone.load_state_dict(state_dict)
    else:
        # Use standard backbones from TorchVision
        backbone = vgg.__dict__[backbone_name](pretrained=pretrained, progress=progress).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorVGG(backbone, highres)

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

    backbone = _vgg_extractor("vgg16_features", False, progress, pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           min_ratio=0.07, max_ratio=0.15)

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (270, 270), num_classes, **kwargs)
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
		   'labels': torch.from_numpy(np.array([1])).float()})
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

train = False
model = my_ssd300_vgg16(num_classes=1)
env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
#mask, bboxes = generate_batch(env, 16)
#print(bboxes)
if train:
	optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
	min_cost = 20000
	while True:
		  optim.zero_grad()
		  mask, bboxes = generate_batch(env, 4)
		  loss = model.forward(torch.from_numpy(mask).float(), bboxes)
		  loss.backward()
		  optim.step()
		  if loss < min_cost:
		  	min_cost = loss
		  	print('Saving model')
		  	torch.save(model.state_dict(), './ssd_test_newAnchors.pth')
else:
	mask, bboxes = generate_batch(env, 1)
	model.load_state_dict(torch.load('./ssd_test_newAnchors.pth', map_location = torch.device('cpu')))
	model.eval()
	preds = model.forward(torch.from_numpy(mask).float())
	print(preds)

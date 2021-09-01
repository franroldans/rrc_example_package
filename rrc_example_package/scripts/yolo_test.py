import torch
import cv2
from rrc_example_package import rearrange_dice_env
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

env = rearrange_dice_env.SimRearrangeDiceEnv(None, rearrange_dice_env.ActionType.POSITION,step_size=1,visualization=False)
env.reset()
camera_observation = env.platform.get_camera_observation(0).cameras[0].image.transpose(2, 0, 1)
cv2.imwrite('camobs.png', camera_observation)
results = model(camera_observation, size=270)
cv2.imwrite('results.png', results)

results.xyxy[0]
results.pandas().xyxy[0]

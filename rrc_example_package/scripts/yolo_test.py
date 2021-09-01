import torch
from PIL import Image
from rrc_example_package import rearrange_dice_env
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

"""env = rearrange_dice_env.RealRobotRearrangeDiceEnv(rearrange_dice_env.ActionType.POSITION,goal= None,step_size=1,)
env.reset()
camera_observation = env.platform.get_camera_observation(0)
camera_observation = camera_observation.cameras[0].image
cv2.imwrite('camobs.png', camera_observation)
camera_observation = camera_observation.transpose(2, 0, 1)"""
camera_observation = Image.load('./camobs.png')
results = model(camera_observation, size=270)
results.save()
#cv2.imwrite('results.png', results)

results.xyxy[0]
results.pandas().xyxy[0]

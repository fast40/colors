from models import DEVICE, ITERATIONS, TargetModel, ChoiceModel
from utils import get_cv2_image
import torch
import cv2

torch.manual_seed(0)

target_model = TargetModel()
choice_model = ChoiceModel()

target_model.load_state_dict(torch.load('target_model_1.pt'))
choice_model.load_state_dict(torch.load('choice_model_1.pt'))

target_model.to(DEVICE)
choice_model.to(DEVICE)

target_model.eval()
choice_model.eval()

flag = False

while not flag:
	decisions = [torch.zeros(1, 6, device=DEVICE)] * ITERATIONS

	for iteration in range(ITERATIONS):
		target_prediction = target_model(decisions)
		print(target_prediction)

		choices = choice_model(decisions)

		image = get_cv2_image(choices, target_prediction)

		cv2.imshow('image', image)

		key = cv2.waitKey(0) & 0xff

		if key == ord('q'):
			flag = True
			break
		elif key == 2:  # left
			print('left')
			decision = choices.clone()
		elif key == 3:  # right
			print('right')
			decision = choices.clone()
			decision[:, :3] = choices[:, :3]
			decision[:, 3:] = choices[:, 3:]
		
		decisions[iteration] = decision

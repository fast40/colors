import torch
import numpy as np

def get_decision(choices, target):
	decision = choices.clone()

	# print('--------------')
	# print(choices.shape)
	# print(target.shape)

	distance1 = ((decision[:, :3] - target) ** 2).sum(dim=1)
	distance2 = ((decision[:, 3:] - target) ** 2).sum(dim=1)

	swap_indices = distance1 > distance2

	decision[swap_indices, :3] = choices[swap_indices, 3:]
	decision[swap_indices, 3:] = choices[swap_indices, :3]

	return decision


def get_cv2_image(choices, target_prediction):
	cv2_image = np.zeros((512, 512, 3), dtype=np.uint8)

	choices = choices.detach().numpy()
	choices = choices.reshape((2, 3))
	choices = choices.clip(0, 255)
	choices = choices.astype(np.uint8)

	choice1 = choices[0].reshape((1, 1, 3))
	choice2 = choices[1].reshape((1, 1, 3))

	target_prediction = target_prediction.detach().numpy()
	target_prediction = target_prediction.reshape((1, 1, 3))
	target_prediction = target_prediction.clip(0, 255)
	target_prediction = target_prediction.astype(np.uint8)

	print(target_prediction)
	target_prediction = target_prediction.repeat(256, axis=0).repeat(512, axis=1)
	choice1 = choice1.repeat(256, axis=0).repeat(256, axis=1)
	choice2 = choice2.repeat(256, axis=0).repeat(256, axis=1)

	cv2_image[0:256, 0:512, :] = target_prediction
	cv2_image[256:, 0:256, :] = choice1
	cv2_image[256:, 256:, :] = choice2

	return cv2_image

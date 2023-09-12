import torch
import numpy as np
import cv2

torch.manual_seed(42)

NUM_EXAMPLES = 25

EVAL_TARGETS = torch.rand((NUM_EXAMPLES, 3))


def display_progress(targets, predictions, size=50):
	image = torch.cat((
		targets.view((1, -1, 3)),
		predictions.view((1, -1, 3))
	)).detach().numpy()

	image[:, :, 0] *= 180
	image[:, :, 1] *= 255
	image[:, :, 2] *= 255

	image = image.clip(0, 255)
	image = image.astype(np.uint8)
	image = image.repeat(size, axis=0).repeat(size, axis=1)

	image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

	cv2.imshow('progress', image)


if __name__ == '__main__':
	predictions = torch.rand((NUM_EXAMPLES, 3))
	display_progress(EVAL_TARGETS, predictions)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
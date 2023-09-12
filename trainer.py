from models import ITERATIONS, TargetModel, ChoiceModel
from utils import get_decision
import torch
from torch.nn import functional as F
from torch import optim
from eval import NUM_EXAMPLES, EVAL_TARGETS, display_progress
import cv2

torch.manual_seed(0)

if torch.cuda.is_available:
	DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
	DEVICE = torch.device('mps')
else:
	DEVICE = torch.device('cpu')

BATCH_SIZE = 1000

target_model = TargetModel()
choice_model = ChoiceModel()

target_model.to(DEVICE)
choice_model.to(DEVICE)

EVAL_TARGETS = EVAL_TARGETS.to(DEVICE)

target_model_optimizer = optim.Adam(target_model.parameters(), lr=0.0001)
choice_model_optimizer = optim.Adam(choice_model.parameters(), lr=0.0001)

losses = []


def run_cycle(targets, train=True):  # TODO: if targets is None enter manual mode
	if train:
		decisions = [torch.zeros((BATCH_SIZE, 6), device=DEVICE)] * ITERATIONS
	else:
		decisions = [torch.zeros((NUM_EXAMPLES, 6), device=DEVICE)] * ITERATIONS

	for iteration in range(ITERATIONS):
		choices = choice_model(decisions)

		decision = get_decision(choices, targets)
		decisions[iteration] = decision

		target_prediction = target_model(decisions)

		if train:
			target_model_optimizer.zero_grad()
			choice_model_optimizer.zero_grad()

			target_loss = F.mse_loss(target_prediction, targets)
			target_loss.backward()

			target_model_optimizer.step()
			choice_model_optimizer.step()

			decisions[iteration] = decisions[iteration].detach()

			if iteration == ITERATIONS - 1:
				return target_prediction, target_loss.item()
	
	return target_prediction
	

try:
	for i in range(10000):
		# print(i)
		targets = torch.rand((BATCH_SIZE, 3), dtype=torch.float, device=DEVICE)

		run_cycle(targets, train=True)
		
		if i % 100 == 99:
			print('update screen')
			predictions = run_cycle(EVAL_TARGETS, train=False)

			display_progress(EVAL_TARGETS, predictions)

			if cv2.waitKey(1) & 0xff == ord('q'):
				quit()
			
			# print(f'{i}: {sum(losses) / len(losses)}')
			# losses = []
finally:
	torch.save(target_model.state_dict(), 'target_model.pt')
	torch.save(choice_model.state_dict(), 'choice_model.pt')

	cv2.destroyAllWindows()
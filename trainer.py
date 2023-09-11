from models import DEVICE, ITERATIONS, TargetModel, ChoiceModel
from utils import get_decision
import torch
from torch.nn import functional as F
from torch import optim

torch.manual_seed(0)

BATCH_SIZE = 1000

target_model = TargetModel()
choice_model = ChoiceModel()

target_model.to(DEVICE)
choice_model.to(DEVICE)

target_model_optimizer = optim.Adam(target_model.parameters(), lr=0.0001)
choice_model_optimizer = optim.Adam(choice_model.parameters(), lr=0.0001)

losses = []

try:
	for i in range(10000):
		target = torch.randint(0, 256, (BATCH_SIZE, 3), dtype=torch.float, device=DEVICE)
		decisions = [torch.zeros(BATCH_SIZE, 6, device=DEVICE)] * ITERATIONS

		for iteration in range(ITERATIONS):
			choices = choice_model(decisions)

			decision = get_decision(choices, target)
			decisions[iteration] = decision

			target_prediction = target_model(decisions)

			target_loss = F.mse_loss(target_prediction, target)

			if iteration == ITERATIONS - 1:
				losses.append(target_loss.item())

			target_model_optimizer.zero_grad()		
			choice_model_optimizer.zero_grad()		

			target_loss.backward()

			target_model_optimizer.step()
			choice_model_optimizer.step()

			decisions[iteration] = decisions[iteration].detach()
		
		if i % 100 == 0:
			print(f'{i}: {sum(losses) / len(losses)}')
			losses = []
finally:
	torch.save(target_model.state_dict(), 'target_model.pt')
	torch.save(choice_model.state_dict(), 'choice_model.pt')
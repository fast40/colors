import torch
from torch import nn
from torch.nn import functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ITERATIONS = 24


class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ITERATIONS * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        concatenated_input = torch.cat(x, dim=1)

        return self.main(concatenated_input)


class ChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ITERATIONS * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        concatenated_input = torch.cat(x, dim=1)

        return self.main(concatenated_input)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# DQN Alg
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float() # Cast to float to fix bug
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.nn.functional.softmax(self.layer3(x), dim=0)

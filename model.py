from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class AC(Module):
    def __init__(self, features_n, actions_n):
        super(AC, self).__init__()
        self.hidden_layer_cells = 256
        self.l1 = nn.Linear(features_n, self.hidden_layer_cells)
        self.actor_linear = nn.Linear(self.hidden_layer_cells, actions_n)
        self.critic_linear = nn.Linear(self.hidden_layer_cells, 1)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        pi = self.actor_linear(x)
        q = self.critic_linear(x)
        return q, pi

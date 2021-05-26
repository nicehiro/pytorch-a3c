from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_init


class AC(Module):
    # def __init__(self, features_n, actions_n):
    #     super(AC, self).__init__()
    #     self.hidden_layer_cells = 256
    #     self.l1 = nn.Linear(features_n, self.hidden_layer_cells)
    #     self.l2 = nn.Linear(self.hidden_layer_cells, self.hidden_layer_cells)
    #     self.actor_linear = nn.Linear(self.hidden_layer_cells, actions_n)
    #     self.critic_linear = nn.Linear(self.hidden_layer_cells, 1)
    #     set_init([self.l1, self.l2, self.actor_linear, self.critic_linear])

    # def forward(self, inputs):
    #     x = torch.tanh(self.l1(inputs))
    #     x = torch.tanh(self.l2(x))
    #     pi = self.actor_linear(x)
    #     q = self.critic_linear(x)
    #     return pi, q

    def __init__(self, s_dim, a_dim):
        super(AC, self).__init__()
        self.hidden_layer_cells = 128
        self.l1 = nn.Linear(s_dim, self.hidden_layer_cells)
        self.l2 = nn.Linear(self.hidden_layer_cells, self.hidden_layer_cells)
        self.actor_linear = nn.Linear(self.hidden_layer_cells, a_dim)
        self.critic_linear = nn.Linear(self.hidden_layer_cells, 1)

    def forward(self, inputs):
        x = torch.tanh(self.l1(inputs))
        x = torch.tanh(self.l2(x))
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values

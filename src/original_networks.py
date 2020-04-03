import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

from src.variables import variables

device = variables["device"]


def make_linear_with_weights(
    in_features: int, out_features: int, weights_range=None
):
    x = nn.Linear(in_features, out_features)

    if not weights_range:
        limit = 1.0 / math.sqrt(in_features)
    else:
        limit = weights_range

    x.weight = nn.Parameter(
        data=Uniform(-limit, limit).sample(x.weight.shape), requires_grad=True
    )
    return x


class Actor(nn.Module):
    def __init__(
        self,
        obs_space_size=8,
        action_space_size=2,
        hidden1=300,
        hidden2=400,
        weights_range=3e-3,
    ):
        super(Actor, self).__init__()
        self.bn_in = nn.BatchNorm1d(num_features=obs_space_size)
        self.input = make_linear_with_weights(obs_space_size, hidden1)
        self.bn_h1 = nn.BatchNorm1d(num_features=hidden1)
        self.hidden1 = make_linear_with_weights(hidden1, hidden2)
        self.bn_out = nn.BatchNorm1d(num_features=hidden2)
        self.output = make_linear_with_weights(
            hidden2, action_space_size, weights_range
        )
        self.to(device)

    def forward(self, x):
        x = F.relu(self.input(self.bn_in(x)))
        x = F.relu(self.hidden1(self.bn_h1(x)))
        x = torch.tanh(self.output(self.bn_out(x)))
        return x


class Critic(nn.Module):
    def __init__(
        self,
        obs_space_size=8,
        action_space_size=2,
        hidden1=300,
        hidden2=400,
        weights_range=3e-3,
    ):
        super(Critic, self).__init__()
        self.bn_in = nn.BatchNorm1d(num_features=obs_space_size)
        self.input = make_linear_with_weights(obs_space_size, hidden1)
        self.bn_h1 = nn.BatchNorm1d(num_features=hidden1)
        self.hidden1 = make_linear_with_weights(hidden1, hidden2)
        self.bn_h2 = nn.BatchNorm1d(num_features=hidden2 + action_space_size)
        self.hidden2 = make_linear_with_weights(
            hidden2 + action_space_size, hidden2
        )
        self.output = make_linear_with_weights(hidden2, 1, weights_range)
        self.to(device)

    def forward(self, x, a):
        x = F.relu(self.input(self.bn_in(x)))
        x = F.relu(self.hidden1(self.bn_h1(x)))
        x = torch.cat((x, a), 1)
        x = F.relu(self.hidden2(self.bn_h2(x)))
        x = self.output(x)
        return x

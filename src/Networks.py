import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

from src.variables import variables

device = variables["device"]


def make_linear_with_weights(in_features: int, out_features: int, weights_range=None):
    x = nn.Linear(in_features, out_features)
    limit = 1.0 / math.sqrt(in_features) if weights_range is None else weights_range
    x.weight = nn.Parameter(data=Uniform(-limit, limit).sample((x.weight.shape)).detach())
    return x


class Actor(nn.Module):
    def __init__(
        self, obs_space_size=8, action_space_size=2, hidden1=300, hidden2=400, weights_range=3e-3,
    ):
        super(Actor, self).__init__()
        self.bn_pre = nn.BatchNorm1d(num_features=obs_space_size)
        self.input = make_linear_with_weights(obs_space_size, hidden1)
        self.bn_in = nn.BatchNorm1d(num_features=hidden1)
        self.hidden1 = make_linear_with_weights(hidden1, hidden2)
        self.bn_h1 = nn.BatchNorm1d(num_features=hidden2)
        self.hidden2 = make_linear_with_weights(hidden2, hidden2)
        self.bn_h2 = nn.BatchNorm1d(num_features=hidden2)
        self.output = make_linear_with_weights(hidden2, action_space_size, weights_range)
        self.to(device)

    def forward(self, x):
        if x.shape[0] > 1:  # check if a batch or a single action is given
            x = F.relu(self.bn_in(self.input(x)))
            x = F.relu(self.bn_h1(self.hidden1(x)))
            x = F.relu(self.bn_h2(self.hidden2(x)))
        else:
            x = self.bn_pre(x)
            x = F.relu(self.input(x))
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
        x = F.tanh(self.output(x))
        return x


class Critic(nn.Module):
    def __init__(
        self, obs_space_size=8, action_space_size=2, hidden1=300, hidden2=400, weights_range=3e-3,
    ):
        super(Critic, self).__init__()
        self.input = make_linear_with_weights(obs_space_size, hidden1)
        self.bn_in = nn.BatchNorm1d(num_features=hidden1)
        self.hidden1 = make_linear_with_weights(hidden1, hidden2)
        self.bn_h1 = nn.BatchNorm1d(num_features=hidden2)
        self.hidden2 = make_linear_with_weights(hidden2 + action_space_size, hidden2)
        self.output = make_linear_with_weights(hidden2, 1, weights_range)
        self.to(device)

    def forward(self, x, a):
        x = F.relu(self.bn_in(self.input(x)))
        x = F.relu(self.bn_h1(self.hidden1(x)))
        # add actions in the second hidden layer as per the paper
        x = torch.cat((x, a), 1)
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x

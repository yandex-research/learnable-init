import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["normal_initializer", "normal_initializer_from_weight", "kaiming_normal_initializer_given_fan",
           "kaiming_normal_weight_initializer", "kaiming_normal_bias_initializer",
           "xavier_normal_weight_initializer", "fixup_resnet_module_weight_initializer"]


class LearnedInitalizer(nn.Module):
    def __init__(self, mean=0.0, std=1.0):

        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=True)  # sqrt for smoother function

    def forward(self, weight):
        device = self.mean.device
        return torch.randn_like(weight, device=device) * self.std + self.mean


def normal_initializer(mean=0.0, std=1.0):
    return LearnedInitalizer(float(mean), float(std))


def normal_initializer_from_weight(weight):
    mean = float(weight.mean())
    std = float(weight.std())
    if mean == std == 0.0:
        return None
    else:
        return LearnedInitalizer(mean, std)


def kaiming_normal_weight_initializer(weight, a=0.01, mode='fan_in', nonlinearity='relu'):
    fan = nn.init._calculate_correct_fan(weight, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return LearnedInitalizer(std=std)


def kaiming_normal_bias_initializer(weight, mode='fan_in'):
    fan = nn.init._calculate_correct_fan(weight, mode)
    std = 1. / math.sqrt(fan)
    return LearnedInitalizer(std=std)


def kaiming_normal_initializer_given_fan(fan):
    std = 1. / math.sqrt(fan)
    return LearnedInitalizer(std=std)


def xavier_normal_weight_initializer(weight, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return LearnedInitalizer(std=std)


#########
# Fixup #
#########


def fixup_resnet_module_weight_initializer(name, module, num_layers):
    if isinstance(module, nn.Conv2d) and 'conv1' == name:
        return normal_initializer_from_weight(module.weight)
    elif isinstance(module, nn.Conv2d) and ('conv1' in name or 'downsample' in name):
        bound = np.sqrt(2 / (module.weight.shape[0] * np.prod(module.weight.shape[2:])))
        if 'conv1' in name:
            bound *= num_layers ** (-0.5)
        return LearnedInitalizer(std=bound)
    else:
        return None
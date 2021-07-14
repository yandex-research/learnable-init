import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["normal_quantile", "uniform_quantile_from_weights", "kaiming_uniform_weight_quantile",
           "kaiming_uniform_bias_quantile", "kaiming_uniform_quantile_given_fan", "xavier_uniform_weight_quantile",
           "constant_quantile", "uniform_quantile",
           "fixup_resnet_module_weight_quantile"]


class LearnedQuantile(nn.Module):
    def __init__(self, num_quantiles, init=torch.distributions.Normal(0.0, 1.0), eps=1e-4):
        super().__init__()
        self.num_quantiles = num_quantiles

        self.inv_softplus = lambda x: torch.log(torch.exp(x) - 1. + 1e-5)
        self.softplus = lambda x: torch.log(torch.exp(x) + 1. - 1e-5)

        quantiles = init.icdf(torch.linspace(eps, 1. - eps, num_quantiles + 1))
        self.v = nn.Parameter(self.inv_softplus(quantiles[1:] - quantiles[:-1]), requires_grad=True)
        self.b = nn.Parameter(quantiles[0], requires_grad=True)

    def forward(self, x):
        return self._forward(x.view(-1)).view(*x.shape)

    def _forward(self, x):
        slopes = self.softplus(self.v)
        # Use bias as a bias for f_0
        cumsum = torch.cumsum(torch.cat([self.b.view(1), slopes]), dim=0)

        # Get i for x
        x_floor = torch.floor(x * self.num_quantiles)
        # f consists of (self.num_quantiles-1) pieces
        x_floor = x_floor.clamp(0, self.num_quantiles - 1)
        ids = x_floor.clamp(0, self.num_quantiles - 1).to(dtype=torch.int64)
        x_floor = x_floor / self.num_quantiles

        # Compute slope_i = grad(f_i) = delta_y / delta_x = slopes * self.num_quantiles
        slope_i = torch.gather(slopes * self.num_quantiles, 0, ids)
        cumsum_by_i = torch.gather(cumsum, 0, ids)
        return cumsum_by_i + slope_i * (x - x_floor)


class ScaledQuantile(nn.Module):
    def __init__(self, learned_quantile, init_scale=1.0):
        super().__init__()
        init_scale = learned_quantile.inv_softplus(torch.tensor(init_scale))
        self.scale = nn.Parameter(init_scale, requires_grad=True)
        self.quantile_func = learned_quantile

    def forward(self, x):
        return self.quantile_func.softplus(self.scale) * self.quantile_func(x)


def normal_quantile(num_quantiles, std=1.0):
    quantile_function = LearnedQuantile(num_quantiles, init=torch.distributions.Normal(0, 1), eps=0.008)
    return ScaledQuantile(quantile_function, init_scale=float(std))


def uniform_quantile(num_quantiles, std=0.5):
    quantile_function = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    return ScaledQuantile(quantile_function, init_scale=float(std))


def uniform_quantile_from_weights(num_quantiles, weights):
    init_distribution = torch.distributions.Uniform(-0.5, 0.5)
    quantile = LearnedQuantile(num_quantiles, init=init_distribution, eps=1e-4)
    scale = weights.max() - weights.min()
    if scale == 0.0:
        return None
    else:
        return ScaledQuantile(quantile, init_scale=float(scale))


def constant_quantile(num_quantiles, constant):
    quantile = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    nn.init.constant_(quantile.b, constant)
    return ScaledQuantile(quantile, init_scale=1e-4)


###########
# Kaiming #
###########

def kaiming_uniform_weight_quantile(num_quantiles, weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu'):
    fan = nn.init._calculate_correct_fan(weight, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    quantile = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    return ScaledQuantile(quantile, init_scale=float(bound))


def kaiming_uniform_bias_quantile(num_quantiles, weight, mode='fan_in'):
    fan = nn.init._calculate_correct_fan(weight, mode)
    bound = 1. / math.sqrt(fan)
    quantile = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    return ScaledQuantile(quantile, init_scale=float(bound))


def kaiming_uniform_quantile_given_fan(num_quantiles, fan):
    bound = 1. / math.sqrt(fan)
    quantile = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    return ScaledQuantile(quantile, init_scale=float(bound))


##########
# Xavier #
##########

def xavier_uniform_weight_quantile(num_quantiles, weight, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    quantile = LearnedQuantile(num_quantiles, init=torch.distributions.Uniform(-1, 1), eps=1e-4)
    return ScaledQuantile(quantile, init_scale=float(bound))


#########
# Fixup #
#########


def fixup_resnet_module_weight_quantile(num_quantiles, name, module, num_layers):
    if isinstance(module, nn.Conv2d) and 'conv1' == name:
        # Basic 'conv1' layer is initilized by kaiming uniform (as in PyTorch)
        return uniform_quantile_from_weights(num_quantiles, module.weight)
    elif isinstance(module, nn.Conv2d) and ('conv1' in name or 'downsample' in name):
        bound = np.sqrt(2 / (module.weight.shape[0] * np.prod(module.weight.shape[2:])))
        if 'conv1' in name:
            bound *= num_layers ** (-0.5)

        quantile_function = LearnedQuantile(num_quantiles, init=torch.distributions.Normal(0, 1), eps=0.008)
        return ScaledQuantile(quantile_function, init_scale=float(bound))
    else:
        return None
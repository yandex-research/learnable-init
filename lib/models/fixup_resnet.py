'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    """ 
        We replace scale and bias parameters with nn.Linear(1, 1, bias=False) for more convenient DIMAML application.
        Note that it changes nothing in model training and evaluation. TODO: make DIMAML handle nn.Parameters as well. 
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Linear(1, 1, bias=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Linear(1, 1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bias2a = nn.Linear(1, 1, bias=False)
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Linear(1, 1, bias=False)

        self.bias2b = nn.Linear(1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride

        nn.init.constant_(self.bias1a.weight, 0)
        nn.init.constant_(self.bias1b.weight, 0)
        nn.init.constant_(self.bias2a.weight, 0)
        nn.init.constant_(self.bias2b.weight, 0)
        nn.init.constant_(self.scale.weight, 1)

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a.weight)
        out = self.relu(out + self.bias1b.weight)

        out = self.conv2(out + self.bias2a.weight)
        out = out * self.scale.weight + self.bias2b.weight
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a.weight)

        out += identity
        out = self.relu(out)
        return out


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, default_init=False):
        """
           :param default_init: if True, uses default pytorch init for layers. 
                                Used only for MetaInit evalution.  
        """
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        # We change kernel_size (7 -> 3) and stride (2 -> 1) to adopt it for CIFAR and Tiny Imagenet datasets
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias1 = nn.Linear(1, 1, bias=False) 
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Linear(1, 1, bias=False) 
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize scales and biases
        nn.init.constant_(self.bias1.weight, 0)
        nn.init.constant_(self.bias2.weight, 0)
        
        if not default_init:
            for m in self.modules():
                if isinstance(m, FixupBasicBlock):
                    bound = np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5)
                    nn.init.normal_(m.conv1.weight, mean=0, std=bound)
                    nn.init.constant_(m.conv2.weight, 0)
                    if m.downsample is not None:
                        bound = np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:])))
                        nn.init.normal_(m.downsample.weight, mean=0, std=bound)

            nn.init.constant_(self.fc.weight, 0)
            nn.init.constant_(self.fc.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1.weight)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2.weight)
        return x


def FixupResNet18(**kwargs):
    """Constructs a Fixup-ResNet-18 model.
    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def FixupResNet34(**kwargs):
    """Constructs a Fixup-ResNet-34 model.
    """
    model = FixupResNet(FixupBasicBlock, [3, 4, 6, 3], **kwargs)
    return model
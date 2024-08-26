import argparse
from typing import Callable
from collections import OrderedDict

import torch
import torch.nn as nn


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', '0', 'no', 'n', 'f'}:
        return False
    elif value.lower() in {'true', '1', 'yes', 'y', 't'}:
        return True
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def module_weight_init(module:nn.Module, initializer:Callable, generator:torch.Generator=None):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.GRU)):
        initializer(module.weight, generator=generator)
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.weight.data.fill_(1)
        if module.bias is not None:
            module.bias.data.fill_(0)
    else:
        pass

def modules_weight_init(modules:OrderedDict, mode:str, generator:torch.Generator=None):
    match mode:
        case "normal":
            initializer = nn.init.normal_
        case "uniform":
            initializer = nn.init.uniform_
        case "xavier_normal":
            initializer = nn.init.xavier_normal_
        case "xavier_uniform":
            initializer = nn.init.xavier_uniform_
        case "kaiming_normal":
            initializer = nn.init.kaiming_normal_
        case "kaiming_uniform":
            initializer = nn.init.kaiming_uniform_
    for block in modules.items():
        if isinstance(block, (nn.Linear, nn.Conv2d, nn.GRU, nn.BatchNorm1d, nn.BatchNorm2d)):
            module_weight_init(module=block, initializer=initializer, generator=generator)
        else:
            for module in block:
                module_weight_init(module=module, initializer=initializer, generator=generator)
            


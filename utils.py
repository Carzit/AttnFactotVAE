import argparse
from typing import Callable, Literal, Union, Optional, List
from collections import OrderedDict

import torch
import torch.nn as nn


def str2bool(value:Union[bool, str]):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in {'false', '0', 'no', 'n', 'f'}:
            return False
        elif value.lower() in {'true', '1', 'yes', 'y', 't'}:
            return True
    else:
        raise argparse.ArgumentTypeError(f'Boolean value or bool like string expected. Get unexpected value {value}, whose type is {type(value)}')

def str2dtype(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return torch.float32
    elif dtype == "FP64":
        return torch.float64
    elif dtype == "FP16":
        return torch.float16
    elif dtype == "BF16":
        return torch.bfloat16
    else:
        raise argparse.ArgumentTypeError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
def str2device(device:Literal["auto", "cpu", "cuda"]) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() == "cuda":
        return torch.device("cuda")
    else:
        raise argparse.ArgumentTypeError(f"Unexpected device `{device}`. dtype must be `cuda`, `cpu` or `auto`.")

def multiLinear(input_size:int, 
                output_size:int, 
                num_layers:int=1, 
                nodes:Optional[List[int]]=None)->nn.Sequential:
    if nodes is None:
        if num_layers == 1:
            return nn.Linear(input_size, output_size)
        else:
            layers = []
            step = (input_size - output_size) // (num_layers - 1)
            for i in range(num_layers):
                in_features = input_size - i * step
                out_features = input_size - (i + 1) * step if i < num_layers - 1 else output_size
                layers.append(nn.Linear(in_features, out_features))
            return nn.Sequential(*layers)
    else:
        if len(nodes) == 1:
            return nn.Sequential(nn.Linear(input_size, nodes[0]),
                                 nn.Linear(nodes[0], output_size))
        else:
            layers = [nn.Linear(input_size, nodes[0])]
            for i in range(len(nodes)):
                layers.append(nn.Linear(nodes[i], nodes[i+1]))
            layers.append(nn.Linear(nodes[-1], output_size))
            return nn.Sequential(*layers)

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
                
def check(tensor:torch.Tensor):
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))
            


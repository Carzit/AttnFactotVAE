import ast
import argparse
from typing import Callable, Literal, Union, Optional, List
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.optim as optim

import lion_pytorch
import diffusers
import transformers


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


def str2dict(args_list):
    result_dict = {}
    if args_list is not None and len(args_list) > 0:
        for arg in args_list:
            key, value = arg.split("=", 1)  # 使用 1 限制分割次数，避免错误处理包含 '=' 的值
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError): # 如果 literal_eval 失败，就把 value 当作字符串处理
                pass
            result_dict[key] = value
    return result_dict
    

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

def get_optimizer(args:argparse.Namespace, trainable_params) -> torch.optim.Optimizer:
    # "Optimizer to use: Adam, AdamW, Lion, SGDNesterov, DAdaptation, Adafactor"

    optimizer_type = args.optimizer_type
    lr = args.learning_rate
    optimizer_kwargs = str2dict(args.optimizer_kwargs)

    if optimizer_type == "Lion":
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("Module `lion_pytorch` not found.")
        print(f"use Lion optimizer | {optimizer_kwargs}")
        """
        lion_pytorch.Lion(
            params,
            lr: 'float' = 0.0001,
            betas: 'Tuple[float, float]' = (0.9, 0.99),
            weight_decay: 'float' = 0.0,
            use_triton: 'bool' = False,
            decoupled_weight_decay: 'bool' = False,
        )
        """
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov":
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(f"SGD with Nesterov must be with momentum, set momentum to 0.9")
            optimizer_kwargs["momentum"] = 0.9
        """
        torch.optim.SGD(
            params,
            lr:float=0.001,
            momentum:float=0,
            dampening:float=0,
            weight_decay:float=0,
            nesterov:bool=True
        )
        """
        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type == "DAdaptation":
        try:
            import dadaptation
        except ImportError:
            raise ImportError("Module `dadaptation` not found.")
        print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")

        actual_lr = lr
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            actual_lr = trainable_params[0].get("lr", actual_lr)
        if actual_lr <= 0.1:
            print(f"learning rate is too low. Learning rate around 1.0 is recommended if using dadaptation: lr={actual_lr}")
        """
        dadaptation.DAdaptAdam(
            params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            log_every=0,
            decouple=False,
            use_bias_correction=False,
            d0=1e-06,
            growth_rate=inf,
            fsdp_in_use=False,
        )
        """
        optimizer_class = dadaptation.DAdaptAdam
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor":
        print(f"use Adafactor optimizer | {optimizer_kwargs}")

        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
            optimizer_kwargs["relative_step"] = True
        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true.")
            if lr != 0.0:
                print(f"Learning rate is used as initial_lr.")
            if args.lr_scheduler_type != "adafactor":
                print(f"Use adafactor_scheduler.")
                args.lr_scheduler_type = f"adafactor"  # ちょっと微妙だけど
            lr = None
        else:
            if args.max_grad_norm != 0.0:
                print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                )
            if args.lr_scheduler != "constant_with_warmup":
                print(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                print(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")
        
        """
        transformers.optimization.Adafactor(
            params,
            lr=None,
            eps=(1e-30, 0.001),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=True,
            relative_step=True,
            warmup_init=False,
        )
        """
        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    elif optimizer_type == "AdamW":
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        """
        torch.optim.AdamW(
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            lr: Union[float, torch.Tensor] = 0.001,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-08,
            weight_decay: float = 0.01,
            amsgrad: bool = False)
        """
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])
    return optimizer


def get_lr_scheduler(args:argparse.Namespace, optimizer:torch.optim.Optimizer, num_processes: int = 1) -> torch.optim.lr_scheduler.LRScheduler:
    # ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"]
    name:str = args.lr_scheduler_type
    num_warmup_steps: int = args.lr_scheduler_warmup_steps # default 0
    num_cycles: int = args.lr_scheduler_num_cycles # default 0.5
    power: int = args.lr_scheduler_power #default 1.0
    num_training_steps: int = args.max_epoches * num_processes * 1 #args.gradient_accumulation_steps
    
    if name.lower() == "constant":
        lr_scheduler = diffusers.optimization.get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif name.lower() == "linear":
        lr_scheduler = diffusers.optimization.get_linear_schedule_with_warmup(optimizer, 
                                                                              num_warmup_steps=num_warmup_steps, 
                                                                              num_training_steps=num_training_steps)
    elif name.lower() == "cosine":
        lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=num_warmup_steps,
                                                                              num_training_steps=num_training_steps, 
                                                                              num_cycles=num_cycles)
    elif name.lower() == "cosine_with_restarts":
        lr_scheduler = diffusers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                                                                 num_warmup_steps=num_warmup_steps, 
                                                                                                 num_training_steps=num_training_steps,
                                                                                                 num_cycles=num_cycles)
    elif name.lower() == "polynomial":
        lr_scheduler = diffusers.optimization.get_polynomial_decay_schedule_with_warmup(optimizer, 
                                                                                        num_warmup_steps=num_warmup_steps,
                                                                                        num_training_steps=num_training_steps,
                                                                                        power=power)
    elif name.lower() == "adafactor":
        assert type(optimizer) == transformers.optimization.Adafactor, f"Adafactor Scheduler must be used with Adafactor Optimizer. Unexpected optimizer type {type(optimizer)}"
        lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=args.lr)
    return lr_scheduler
        

            


import os
import random
from datetime import datetime


import numpy as np
import torch


def time_str():
    return datetime.now().strftime('[%m-%d %H:%M:%S]')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_params(model: torch.nn.Module, verbose=False):
    if verbose:
        print('\n===================== param initialization begins =====================')
    tot_num_inited = 0
    for i, m in enumerate(model.modules()):
        clz = m.__class__.__name__
        is_conv = clz.find('Conv') != -1
        is_bn = clz.find('BatchNorm') != -1
        is_fc = clz.find('Linear') != -1
        
        cur_num_inited = []
        if is_conv:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_bn:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_fc:
            # torch.nn.init.normal_(m.weight, std=0.001)
            torch.nn.init.normal_(m.weight, std=1 / m.weight.size(-1))
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        tot_num_inited += sum(cur_num_inited)
        
        if verbose:
            builtin = any((is_conv, is_bn, is_fc))
            cur_num_inited = f' ({" + ".join([str(x) for x in cur_num_inited])})'
            print(f'clz{i:3d}: {"  => " if builtin else ""}{clz}{cur_num_inited if builtin else "*"}')
    
    if verbose:
        print('----------------------------------------------------------------')
        print(f'tot_num_inited: {tot_num_inited}')
        print('====================== param initialization ends ======================\n')
    return tot_num_inited

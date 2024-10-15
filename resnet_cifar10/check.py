import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import torch.backends.cudnn as cudnn
import argparse
import datetime
import os
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', 
        help='Directory for load weight.')

args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path = args.path
print('path :', path)

print(torch.cuda.is_available())

total=0
#weights = torch.load(path)
#state_dict = weights['state_dict']
state_dict = torch.load(path)

#print("module.layer1.4.conv1.weight[2]\n", state_dict['module.layer1.4.conv1.weight'][2])


for key in list(state_dict.keys()):
    if key.startswith('module.layer') and (key.find('conv')>0):
        l1_norm = state_dict[key].reshape(-1, 9).abs().sum(dim=1)
        var = l1_norm.var(dim=0)
        print(key,'\t\t', var)

print()
print('Finished')
print()
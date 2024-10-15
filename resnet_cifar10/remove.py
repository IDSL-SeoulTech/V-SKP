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
parser.add_argument('--save', default='weights/', 
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
state_dict = torch.load(path)
weight = {} 

for key in list(state_dict.keys()):
    weight[key[7:]] = state_dict[key]

torch.save(weight, os.path.join(args.save, 'cpu_resnet56.pth'))
print()
print('Finished')
print()
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
parser.add_argument('--path2', default='weights/', 
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


total = 0
flops = 0

remained_total = 0
remained_flops = 0

checkpoint = torch.load(args.path)
state_dict = torch.load(args.path2)

for key in list(checkpoint.keys()):
    if (key.find('bn')>0) or (key.find('bias')>0):
        pass
    else:
        if key.startswith('module.conv1.weight'):
            total = checkpoint[key].view(-1).shape[0]
            flops += total * 32 * 32
        elif key.startswith('module.layer1.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            total = checkpoint[key].view(-1).shape[0]
            flops += (total) * 32 * 32
        elif key.startswith('module.layer2.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            total = checkpoint[key].view(-1).shape[0]
            flops += (total) * 16 * 16
        elif key.startswith('module.layer3.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            total = checkpoint[key].view(-1).shape[0]
            flops += (total) * 8 * 8
        elif key.startswith('module.linear.weight'):
            total = checkpoint[key].view(-1).shape[0]
            flops += (total)



for key in list(state_dict.keys()):
    if (key.find('bn')>0) or (key.find('bias')>0):
        pass
    else:
        if key.startswith('module.conv1.weight'):
            remained_total = state_dict[key].view(-1).shape[0]
            remained_flops += remained_total * 32 * 32
        elif key.startswith('module.layer1.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            remained_total = state_dict[key].view(-1).shape[0]
            remained_flops += (remained_total) * 32 * 32
        elif key.startswith('module.layer2.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            remained_total = state_dict[key].view(-1).shape[0]
            remained_flops += (remained_total) * 16 * 16
        elif key.startswith('module.layer3.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
            remained_total = state_dict[key].view(-1).shape[0]
            remained_flops += (remained_total) * 8 * 8
        elif key.startswith('module.linear.weight'):
            remained_total = state_dict[key].view(-1).shape[0]
            remained_flops += (remained_total)



print()   
print('total FLOPs :', flops)
print('remnained FLOPs :', remained_flops)
print('pruned ratio :', '%.1f' %((1-(remained_flops/flops))*100), '%')
  

print()
print('Finished')
print()
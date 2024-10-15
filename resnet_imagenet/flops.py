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

parser.add_argument('--path', default='weights/', help='Directory for load weight.')

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

checkpoint = torch.load(path)

total = 0
actual = 0
temp = 0
flops = 0
total_flops = 0

for key in list(checkpoint.keys()):
    if (key.find('bn')>0) or (key.find('bias')>0):
        pass
    else:
        if key.startswith('module.conv1.weight'):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = (total - temp) * 112 * 112
            flops += actual
            total_flops += (total) * 112 * 112
        elif (key.startswith('module.layer1.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'))) or key.startswith('module.layer2.0.conv1.weight'):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = (total - temp) * 56 * 56
            flops += actual
            total_flops += (total) * 56 * 56
        elif (key.startswith('module.layer2.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'))) or key.startswith('module.layer3.0.conv1.weight') and (key.find('layer2.0.conv1.weight')<0):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = (total - temp) * 28 * 28
            flops += actual
            total_flops += (total) * 28 * 28
        elif (key.startswith('module.layer3.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'))) or key.startswith('module.layer4.0.conv1.weight') and (key.find('layer3.conv1.weight')<0):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = (total - temp) * 14 * 14
            flops += actual
            total_flops += (total) * 14 * 14
        elif (key.startswith('module.layer4.') and (key.endswith('conv1.weight') or key.endswith('conv2.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'))) or key.startswith('module.layer4.0.conv1.weight'):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = (total - temp) * 7 * 7
            flops += actual
            total_flops += (total) * 7 * 7
        elif key.startswith('module.fc.weight'):
            total = checkpoint[key].view(-1).shape[0]
            temp = (checkpoint[key].view(-1) == 0).sum()
            actual = total - temp
            flops += actual
            total_flops += (total)

print()   
print('total FLOPs :', total_flops)
print('remained FLOPs :', int(flops))
print('pruned ratio :', '%.1f' %((1-(flops/total_flops))*100), '%')
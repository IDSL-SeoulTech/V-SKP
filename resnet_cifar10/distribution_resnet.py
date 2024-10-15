import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

import torch.backends.cudnn as cudnn
import argparse
import datetime
import os
from torch.autograd import Variable
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='figure/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')
#parser.add_argument('--path2', default='weights/', help='Directory for load weight.')
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

state_dict=torch.load(args.path)
total=0
for key in list(state_dict.keys()):
  if key.startswith('module.layer') and (key.find('conv')>0) and key.endswith('weight'):
    if state_dict[key].shape[-1]==3:
      non_idx = state_dict[key].view(-1,9).abs().sum(dim=1).nonzero().reshape(-1)
      total += non_idx.shape[-1]

data = torch.zeros(total)
index = 0
for key in list(state_dict.keys()):
  if key.startswith('module.layer') and (key.find('conv')>0) and key.endswith('weight'):
    if state_dict[key].shape[-1]==3:
      non_idx = state_dict[key].view(-1,9).abs().sum(dim=1).nonzero().reshape(-1)
      temp_weight = state_dict[key].view(-1,9).abs().sum(dim=1)
      temp_weight = temp_weight[non_idx]
      size = non_idx.shape[-1]
      data[index:(index+size)] = temp_weight
      index += size
sorted_data, _ = data.sort()
data_mean = data.mean().numpy()
data_max = data.max().numpy()
data = data.numpy()
print('data mean :', data_mean)
print('data max :', data_max)
#data_max = int(data_max.ceil())
print('data_max ceil :', data_max)
plt.hist(data, alpha = 1.0, bins = 500, range = [0.000, 1.2], label='3by3 Conv L1 Norm')
plt.ylim([0,1500])

#plt.hist(data, alpha = 1.0, bins = 500, range = [0.000, 1.5], label='3by3 Conv L1 Norm', cumulative='True')
'''
plt.axvline(sorted_data[int(data.shape[0]*0.1)], color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(sorted_data[int(data.shape[0]*0.2)], color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(sorted_data[int(data.shape[0]*0.3)], color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(sorted_data[int(data.shape[0]*0.4)], color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(sorted_data[int(data.shape[0]*0.5)], color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(sorted_data[int(data.shape[0]*0.6)], color='red', linestyle='dashed', linewidth=1.5)
'''
#plt.axvline(data_mean, color='green', linestyle='dashed', linewidth=1.5)#, label = 'Avg scaling factor with BST(0.1751)')
#plt.axvline(0.2424, color='blue', linestyle='dashed', linewidth=1.5)#, label = 'Avg scaling factor of pre-trained(0.2424)')
#plt.axvline(data2_mean, color='red', linestyle='dashed', linewidth=1.5)#, label = 'Avg scaling factor with SST(0.2600)')
plt.legend()
plt.savefig(os.path.join(args.save, 'weight_distribution.png'))

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
parser.add_argument('--mode', default=2, type=int)

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

total_3x3 = 0
total_1x1 = 0
total = 0
actual_3x3 = 0
actual_1x1 = 0
actual = 0
state_dict = torch.load(path)

#Only Downsample
if args.mode==1:
  for key in list(state_dict.keys()):
    if key.endswith('downsample.0.weight'):
      total_1x1 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1) == 0)
        actual_1x1 += mask.sum()
    
    elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
      total_3x3 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1,9).sum(dim=1) == 0)
        actual_3x3 += mask.sum()
#Whole Pruning          
elif args.mode==2:
  for key in list(state_dict.keys()):
    if key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0)) and (key.find('weight')>0):
      total_1x1 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1) == 0)
        actual_1x1 += mask.sum()
      
    
    elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
      total_3x3 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1,9).sum(dim=1) == 0)
        actual_3x3 += mask.sum()

#Downsample
else:
  for key in list(state_dict.keys()):
    if key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0) or key.endswith('downsample.0.weight')) and (key.find('weight')>0):
      total_1x1 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1) == 0)
        actual_1x1 += mask.sum()
    
    elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
      total_3x3 += state_dict[key].shape[0] * state_dict[key].shape[1]
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      for i in range(state_dict[key].shape[0]):
        mask = (state_dict[key][i].view(-1,9).sum(dim=1) == 0)
        actual_3x3 += mask.sum()

  

actual_1x1 = int(actual_1x1)
actual_3x3 = int(actual_3x3)
actual = int(actual)
total = int(total)
print('total 3by3 :', total_3x3)
print('total 1by1 :', total_1x1)
print('total :', total)

print('actual 1by1 :', actual_1x1, '\tpruning ratio :', '%.1f' %((actual_1x1/total_1x1)*100),'%')
print('actual 3by3 :', actual_3x3, '\tpruning ratio :', '%.1f' %((actual_3x3/total_3x3)*100),'%')
print('actual :', actual, '\tpruning ratio :', '%.1f' %((actual/total)*100),'%')
print()
print('Finished')
print()
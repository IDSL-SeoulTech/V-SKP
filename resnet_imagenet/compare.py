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

parser.add_argument('--path1', default='weights/', help='Directory for load weight.')
parser.add_argument('--path2', default='weights/', help='Directory for load weight.')

args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path1 = args.path1
path2 = args.path2
print('path1 :', path1)
print('path2 :', path2)

print(torch.cuda.is_available())

total1_3x3 = 0
total1_1x1 = 0
total2_3x3 = 0
total2_1x1 = 0
total1 = 0
total2 = 0
state_dict = torch.load(path1)
weight = torch.load(path2)

for key in list(state_dict.keys()):
  if key.endswith('num_batches_tracked'):
    pass
  else:
    total1 += state_dict[key].view(-1).shape[0]
    
for key in list(weight.keys()):
  if key.endswith('num_batches_tracked'):
    pass
  else:
    total2 += weight[key].view(-1).shape[0]

total1 = int(total1)
total2 = int(total2)
print('PATH1 parameters num :', total1)
print('PATH2 parameters num :', total2)

if total1>=total2:
  print('Pruned Rate :', '%.1f' %((1-(total2/total1))*100), '%') 
else:
  print('Pruned Rate :', '%.1f' %((1-(total1/total2))*100), '%')
  

print()
print('Finished')
print()
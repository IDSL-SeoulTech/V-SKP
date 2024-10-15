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
parser.add_argument('--mode', default=0, type=int)

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
actual_3x3 = 0
actual_1x1 = 0
total = 0
actual = 0
state_dict = torch.load(path)


#Except BN Parameters
if args.mode==1:
  for key in list(state_dict.keys()):
    if (key.find('bn')>0) or (key.find('bias')>0):
      pass
    else:
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
      print(key, float((state_dict[key].view(-1) == 0).sum() / state_dict[key].view(-1).shape[0])*100,'%')
  
  print('total parameters num :', total)
  print('remained paramters num :', int(total - actual))
  print('pruned ratio :', '%.1f' %((actual/total)*100), '%')

#Whole Parameters
else:
  for key in list(state_dict.keys()):
    if key.endswith('num_batches_tracked'):
      pass
    else:
      total += state_dict[key].view(-1).shape[0]
      actual += (state_dict[key].view(-1) == 0).sum()
  
  print('total parameters num :', total)
  print('remained parameters num :', int(total - actual))
  print('pruned ratio :', '%.1f' %((actual/total)*100), '%')

  

print()
print('Finished')
print()
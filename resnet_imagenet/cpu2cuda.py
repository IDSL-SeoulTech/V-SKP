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
import time

parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='weights/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')

args = parser.parse_args()

torch.cuda.current_device()

#device = 'cpu'

use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path = args.path
print('path :', path)

print(torch.cuda.is_available())


state_dict = torch.load(path)

with torch.no_grad():
  for key in list(state_dict.keys()):
    state_dict[key] = state_dict[key].cuda()

torch.save(state_dict, os.path.join(args.save, 'resnet50_Baseline.pth')) 

print('CPU TO CUDA')

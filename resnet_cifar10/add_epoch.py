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
parser.add_argument('--save', dest='save', help='The directory used to save the trained models', default='weights', type=str)

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

epoch_state = {'epoch' : 0, 'weight' : state_dict}

torch.save(epoch_state, os.path.join(args.save, str(epoch_state['epoch'])+'_ep_trained.pth'))

print()
print('Finished')
print()
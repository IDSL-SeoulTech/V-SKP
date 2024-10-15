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
parser.add_argument('--mode', default=0, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')

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

if args.mode==1:
    for key in list(state_dict.keys()):
        if key.startswith('module.layer') and ((key.find('conv')>0) or  key.endswith('downsample.0.weight')):
            if state_dict[key].shape[3]==1:
                layer_kernel = state_dict[key].reshape(-1).abs()
                layer_var = torch.var(layer_kernel)
                if key.endswith('downsample.0.weight'):    
                    print(key, '\t%.6f' %(float(layer_var)))
                else:
                    print(key, '\t\t%.6f' %(float(layer_var)))

elif args.mode==2:
    for key in list(state_dict.keys()):
      if key.startswith('module.layer') and (key.find('conv')>0) and key.endswith('weight'):
        total += state_dict[key].view(-1,9).shape[0] 
    val = torch.zeros(total)
    idx = 0
    for key in list(state_dict.keys()):
      if key.startswith('module.layer') and (key.find('conv')>0) and key.endswith('weight'):
        size = state_dict[key].view(-1,9).shape[0]
        val[idx:idx+size] = state_dict[key].view(-1,9).abs().sum(dim=1)
        idx += size
        
    norm, _ = torch.sort(val)
    
    for i in range(1,10):
      val_idx = int(total * i * 0.1)
      print('L1 Norm',i*10,'% :', norm[val_idx])
      


elif args.mode==3:
    for key in list(state_dict.keys()):
        if key.startswith('module.layer') and ((key.find('conv')>0) or  key.endswith('downsample.0.weight')):
            if state_dict[key].shape[3]==3:
                layer_kernel = state_dict[key].reshape(-1,9).abs().sum(dim=1)
                layer_var = torch.var(layer_kernel)
                print(key, '\t%.4f' %(float(layer_var)))
                
                
else:
    for key in list(state_dict.keys()):
        if key.startswith('module.layer') and ((key.find('conv')>0) or  key.endswith('downsample.0.weight')):
            if state_dict[key].shape[3]==3:
                layer_kernel = state_dict[key].reshape(-1,9).abs().sum(dim=1)
                layer_var = torch.var(layer_kernel)
                print(key, '\t\t%.6f' %(float(layer_var)))
                
            elif state_dict[key].shape[3]==1:
                layer_kernel = state_dict[key].reshape(-1).abs()
                layer_var = torch.var(layer_kernel)
                if key.endswith('downsample.0.weight'):                    
                    print(key, '\t%.6f' %(float(layer_var)))
                else:
                    print(key, '\t\t%.6f' %(float(layer_var)))
                  

print()
print('Finished')
print()
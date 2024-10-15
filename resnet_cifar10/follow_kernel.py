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
print('path1 :', path1)
path2 = args.path2
print('path2 :', path2)

print(torch.cuda.is_available())

total=0
state_dict1 = torch.load(path1)
state_dict2 = torch.load(path2)

cos = nn.CosineSimilarity(dim=0, eps=1e-6)

print()
#Compare L1 Norm, Variance, Cosine Similarity
name = []
for key in list(state_dict1.keys()):
    if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
        name.append(key)

for key in list(state_dict1.keys()):
    if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
        weight1 = state_dict1[key].clone()
        weight2 = state_dict2[key].clone()
        print()
        print('weight1', key,' L1 Norm {:.2f}'.format(float(weight1.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(weight1.reshape(-1,9).var(dim=1).sum())))
        print('\nweight2', key,' L1 Norm {:.2f}'.format(float(weight2.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(weight2.reshape(-1,9).var(dim=1).sum())))
        print()
        print('Cosine Similarity')
        print(key)
        cos_idx = []
        for i in range(weight1.shape[0]):
            cos_val = cos(weight1[i].reshape(-1), weight2[i].reshape(-1))
            cos_bool = (((cos_val < 1.0001) * cos_val) > 0.9999)
            if cos_bool>0:
                #print(cos_val)
                cos_idx.append(i)
                #print(i,'th filter')
        print(len(cos_idx),' filters are stucked')
        print(cos_idx)
        name_idx = name.index(key)
        kernel_term_weight = state_dict1[name[name_idx+1]].permute(1,0,2,3)       
        find_zero = kernel_term_weight.reshape(kernel_term_weight.shape[0], -1).sum(dim=1) == 0
        zero_idx = find_zero.nonzero().reshape(-1)
        print(zero_idx.shape[0],' kernel term filter were removed')
        print(zero_idx)
        
        
print()
print('Finished')
print()
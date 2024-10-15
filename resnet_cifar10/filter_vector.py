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
for key in list(state_dict1.keys()):
    if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
        weight1 = state_dict1[key].clone()
        weight2 = state_dict2[key].clone()
        total = state_dict1[key].view(-1).shape[0]
        actual = (state_dict1[key].view(-1) == 0).sum()
        retained = (total - actual)/9
        print()
        print('weight1', key,' L1 Norm {:.2f}'.format(float(weight1.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(weight1.reshape(-1,9).var(dim=1).sum())))
        print('weight1 kpL1 : {:.2f}'.format(float(weight1.abs().reshape(-1).sum(dim=0))/(retained)), 'kpV : {:.8f}'.format(float(weight1.reshape(-1,9).var(dim=1).sum())/retained))
        print('weight2', key,' L1 Norm {:.2f}'.format(float(weight2.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(weight2.reshape(-1,9).var(dim=1).sum())))
        print('weight2 kpL1 : {:.2f}'.format(float(weight2.abs().reshape(-1).sum(dim=0))/(total/9)), 'kpV : {:.8f}'.format(float(weight2.reshape(-1,9).var(dim=1).sum())/(total/9)))
        pruned_rate = int(actual/total*1000)/10.0
        print('Cosine Similarity \tpruned rate :', pruned_rate,'%\t total :',total,'actual:',int(actual))
        cos_val = cos(weight1.reshape(-1), weight2.reshape(-1))
        print(key, cos_val)
        


print()
print('Finished')
print()
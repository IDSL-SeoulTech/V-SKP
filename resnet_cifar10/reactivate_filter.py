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

parser.add_argument('--p_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--f_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--o_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')

args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

p_path = args.p_path
print('p_path :', p_path)
f_path = args.f_path
print('f_path :', f_path)
o_path = args.o_path
print('o_path :', o_path)

print(torch.cuda.is_available())

total=0
pruned_weight = torch.load(p_path)
finetuned_weight = torch.load(f_path)
origin_weight = torch.load(o_path)

cos = nn.CosineSimilarity(dim=0, eps=1e-6)

print()

name = []
for key in list(finetuned_weight.keys()):
    if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
        name.append(key)
    
for key in list(pruned_weight.keys()):
    if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
        p_weight = pruned_weight[key].clone()
        f_weight = finetuned_weight[key].clone()
        print()
        print('pruned_weight', key,' L1 Norm {:.2f}'.format(float(p_weight.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(p_weight.reshape(-1,9).var(dim=1).sum())))
        print('\nfinetuned_weight', key,' L1 Norm {:.2f}'.format(float(f_weight.abs().reshape(-1).sum(dim=0))),'Variance {:.2f}'.format(float(f_weight.reshape(-1,9).var(dim=1).sum())))
        print()
        print('Cosine Similarity')
        print(key)
        cos_idx = []
        for i in range(p_weight.shape[0]):
            cos_val = cos(p_weight[i].reshape(-1), f_weight[i].reshape(-1))
            cos_bool = (((cos_val < 1.0001) * cos_val) > 0.9999)
            if cos_bool>0:
                print(cos_val)
                cos_idx.append(i) 
                print(i,'th filter')
              
        
        if len(cos_idx)>0:
            name_idx = name.index(key)
            origin = origin_weight[name[name_idx+1]].clone()
            origin = origin.permute(1,0,2,3)
            ft_weight = finetuned_weight[name[name_idx+1]].permute(1,0,2,3).clone()
            final_idx = []
            for i in range(len(cos_idx)):
                reactivate_kernel = ft_weight[cos_idx[i]].abs().reshape(-1,9).sum(dim=1)
                react_filter = reactivate_kernel.argmax()
                final_idx.append([cos_idx[i], react_filter])
                align_weight = finetuned_weight[name[name_idx+1]][react_filter].abs().reshape(-1,9).sum(dim=1)
                _, idx_val = torch.sort(align_weight)
                zero_check = (align_weight == 0).sum()
                finetuned_weight[name[name_idx+1]][react_filter,idx_val[zero_check]] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                        
                for i in range(len(final_idx)):
                    finetuned_weight[name[name_idx+1]][final_idx[i][1], final_idx[i][0]] = origin[final_idx[i][0],final_idx[i][1]]
                    pruned_weight[name[name_idx+1]][final_idx[i][1], final_idx[i][0]] = origin[final_idx[i][0],final_idx[i][1]]

total = 0
actual = 0
for key in list(finetuned_weight.keys()):
    if key.endswith('num_batches_tracked'):
        pass
    else:
        total += finetuned_weight[key].view(-1).shape[0]
        actual += (finetuned_weight[key].view(-1) == 0).sum()
  
print('total parameters num :', total)
print('remained parameters num :', int(total - actual))
ratio = int(actual/total*1000)/10.0
print('pruned ratio :', ratio, '%')


torch.save(finetuned_weight, os.path.join(args.save, 'ft_ResNet56_'+str(ratio)+'%_pruned.pth'))
torch.save(pruned_weight, os.path.join(args.save, 'woft_ResNet56_'+str(ratio)+'%_pruned.pth'))
        
print()
print('Finished')
print()
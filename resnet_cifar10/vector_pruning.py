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
parser.add_argument('--ratio', default=0.1, type=float, help='Ratio for pruning weights.')
parser.add_argument('--alpha', default=0.95, type=float, help='Ratio for pruning weights.')
parser.add_argument('--beta', default=0.25, type=float, help='Ratio for pruning weights.')
parser.add_argument('--mode', default=0, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH', help='path to save pruned model (default : weights/')

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
weight = torch.load(path)
beta = args.beta
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

print()

#L1 Norm and cos sim
if args.mode==0:
    name = []
    total = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            name.append(key)
            total += weight[key].shape[0] * weight[key].shape[1]
    
    sens = torch.zeros(total)
    acc_cnt = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            for i in range(weight[key].shape[0]):
                cnt = weight[key].shape[1]
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                sens[acc_cnt:(acc_cnt+cnt)] = imp_score
                acc_cnt += cnt
    y, _ = torch.sort(sens)
    thre_index = int(total * (args.ratio))
    thre = y[thre_index].cuda()
    print('\nthre ', thre)
    
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            zero_sum = 0
            for i in range(weight[key].shape[0]):
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                
                mask = torch.tensor(imp_score.gt(thre))
                mask_idx = mask.nonzero()
                zero_sum += torch.sum(mask)
            
            ratio = 1 - zero_sum/float(weight[key].shape[0] * weight[key].shape[1])            
            rv_idx = int(weight[key].shape[1]*ratio)                
            if (ratio == 0):
                pass
            else :
                for i in range(weight[key].shape[0]):
                    filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                    l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                    alpha = args.alpha
                    imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    weight[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    
    torch.save(weight, os.path.join(args.save, 'L1_ResNet56_'+str(args.ratio*100)+'%_pruned.pth'))
    
#L2 Norm and cos sim
elif args.mode==1:
    name = []
    total = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            name.append(key)
            total += weight[key].shape[0] * weight[key].shape[1]
    
    sens = torch.zeros(total)
    acc_cnt = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            for i in range(weight[key].shape[0]):
                cnt = weight[key].shape[1]
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l2_norm = weight[key][i].reshape(weight[key].shape[1], -1)
                l2_norm = torch.sqrt((l2_norm * l2_norm).sum(dim=1))
                alpha = args.alpha
                imp_score = l2_norm * (alpha + cos_val.abs() * (1 - alpha))
                sens[acc_cnt:(acc_cnt+cnt)] = imp_score
                acc_cnt += cnt
    y, _ = torch.sort(sens)
    thre_index = int(total * (args.ratio))
    thre = y[thre_index].cuda()
    print('\nthre ', thre)
    
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            zero_sum = 0
            for i in range(weight[key].shape[0]):
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l2_norm = weight[key][i].reshape(weight[key].shape[1], -1)
                l2_norm = torch.sqrt((l2_norm * l2_norm).sum(dim=1))
                alpha = args.alpha
                imp_score = l2_norm * (alpha + cos_val.abs() * (1 - alpha))
                
                mask = torch.tensor(imp_score.gt(thre))
                mask_idx = mask.nonzero()
                zero_sum += torch.sum(mask)
            
            ratio = 1 - zero_sum/float(weight[key].shape[0] * weight[key].shape[1])            
            rv_idx = int(weight[key].shape[1]*ratio)                
            if (ratio == 0):
                pass
            else :
                for i in range(weight[key].shape[0]):
                    filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                    l2_norm = weight[key][i].reshape(weight[key].shape[1], -1)
                    l2_norm = torch.sqrt((l2_norm * l2_norm).sum(dim=1))
                    alpha = args.alpha
                    imp_score = l2_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    weight[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    
    torch.save(weight, os.path.join(args.save, 'L2_ResNet56_'+str(args.ratio*100)+'%_pruned.pth')) 
    
#L1 Norm + Vector + Variance
elif args.mode==2:
    name = []
    total = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            name.append(key)
            total += weight[key].shape[0] * weight[key].shape[1]
    
    sens = torch.zeros(total)
    acc_cnt = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            layer_norm = weight[key].reshape(-1,9).abs().sum(dim=1)
            for i in range(weight[key].shape[0]):
                cnt = weight[key].shape[1]
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                var_score = torch.var(layer_norm)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * (1 - beta * var_score)
                sens[acc_cnt:(acc_cnt+cnt)] = imp_score
                acc_cnt += cnt
    y, _ = torch.sort(sens)
    thre_index = int(total * (args.ratio))
    thre = y[thre_index].cuda()
    print('\nthre ', thre)
    
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            zero_sum = 0
            layer_norm = weight[key].reshape(-1,9).abs().sum(dim=1)
            for i in range(weight[key].shape[0]):
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                var_score = torch.var(layer_norm)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * (1 - beta * var_score)
                
                mask = torch.tensor(imp_score.gt(thre))
                mask_idx = mask.nonzero()
                zero_sum += torch.sum(mask)
            
            ratio = 1 - zero_sum/float(weight[key].shape[0] * weight[key].shape[1])            
            rv_idx = int(weight[key].shape[1]*ratio)                
            if (ratio == 0):
                pass
            else :
                for i in range(weight[key].shape[0]):
                    filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                    l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                    alpha = args.alpha
                    imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    weight[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    
    torch.save(weight, os.path.join(args.save, 'Var_ResNet56_'+str(args.ratio*100)+'%_a'+str(args.alpha)+'_b'+str(args.beta)+'_pruned.pth'))
    

#L1 Norm + Vector
elif args.mode==3:
    name = []
    total = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            name.append(key)
            total += weight[key].shape[0] * weight[key].shape[1]
    
    sens = torch.zeros(total)
    acc_cnt = 0
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            layer_norm = weight[key].reshape(-1,9).abs().sum(dim=1)
            for i in range(weight[key].shape[0]):
                cnt = weight[key].shape[1]
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                sens[acc_cnt:(acc_cnt+cnt)] = imp_score
                acc_cnt += cnt
    y, _ = torch.sort(sens)
    thre_index = int(total * (args.ratio))
    thre = y[thre_index].cuda()
    print('\nthre ', thre)
    
    for key in list(weight.keys()):
        if (key.endswith('conv1.weight') or key.endswith('conv2.weight')) and key.startswith('module.layer'):
            zero_sum = 0
            layer_norm = weight[key].reshape(-1,9).abs().sum(dim=1)
            for i in range(weight[key].shape[0]):
                filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                
                mask = torch.tensor(imp_score.gt(thre))
                mask_idx = mask.nonzero()
                zero_sum += torch.sum(mask)
            
            ratio = 1 - zero_sum/float(weight[key].shape[0] * weight[key].shape[1])            
            rv_idx = int(weight[key].shape[1]*ratio)                
            if (ratio == 0):
                pass
            else :
                for i in range(weight[key].shape[0]):
                    filter_vector = weight[key][i].reshape(weight[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(weight[key].shape[1], weight[key].shape[2] * weight[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, weight[key][i].reshape(weight[key].shape[1], -1))
                    l1_norm = weight[key][i].reshape(weight[key].shape[1], -1).abs().sum(dim=1)
                    alpha = args.alpha
                    imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    weight[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    
    torch.save(weight, os.path.join(args.save, 'Vec_ResNet56_'+str(args.ratio*100)+'%'+'_a_'+str(args.alpha)+'_pruned.pth'))
    
 

print()
print('Finished')
print()
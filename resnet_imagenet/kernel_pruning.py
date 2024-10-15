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
parser.add_argument('--ratio', default=0.1, type=float, help='Ratio for pruning weights.')
parser.add_argument('--alpha', default=0.85, type=float, help='Ratio for pruning weights.')
parser.add_argument('--beta', default=0.75, type=float, help='Ratio for pruning weights.')
parser.add_argument('--scale', default=200, type=float, help='Ratio for pruning weights.')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')
parser.add_argument('--mode', default=1, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')
parser.add_argument('--per', default=0.001, type=float, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')

args = parser.parse_args()

torch.cuda.current_device()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path = args.path
print('path :', path)

print(torch.cuda.is_available())

class Prune(nn.Module):
    def __init__(self):
        print('Kernel Pruning Process Start!')
    def load_weights(self, path):
        '''Loads weights from a compressed save file.'''
        total=0
        total2=0
        state_dict = torch.load(path)
        beta = args.beta
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        name = []
        for key in list(state_dict.keys()):
          if key.startswith('module.layer'):
            name.append(key)
        
        #3by3 L1 Norm + Vector + Layer Variance
        #1by1 L1 Norm    
        if args.mode==1:
          #3by3 Conv total
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              layer_norm = state_dict[key].reshape(-1,9).abs().sum(dim=1)
              for i in range(state_dict[key].shape[0]):
                size = state_dict[key].shape[1]
                filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                var_score = torch.var(layer_norm)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * (1 - beta * var_score)
                cv[index:(index+size)] = imp_score
                index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          #1by1 Conv L1 Norm + Layer Variance
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              size = state_dict[key].view(-1).shape[0]
              l1_norm = state_dict[key].view(-1).abs()
              _, k_idx = torch.sort(l1_norm)
              keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
              layer_var = torch.var(l1_norm) * args.scale
              l1_norm[keep_kernel] = 1000
              norm[index2:(index2+size)] = l1_norm * (1 - beta * layer_var)
              index2 += size
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('weight'):
              if key.endswith('conv2.weight'):
                print(key)
                zero_sum = 0
                layer_norm = state_dict[key].reshape(-1,9).abs().sum(dim=1)
                var_score = torch.var(layer_norm)
                for i in range(state_dict[key].shape[0]):
                  filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                  stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                  stand_vector[:] = filter_vector
                  cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                  l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                  alpha = args.alpha
                  imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * (1 - beta * var_score)
                  mask = torch.tensor(imp_score.gt(thre))
                  mask_idx = mask.nonzero()
                  zero_sum += torch.sum(mask)
                ratio = 1 - zero_sum/float(state_dict[key].shape[0] * state_dict[key].shape[1])            
                rv_idx = int(state_dict[key].shape[1]*ratio)                
                if (ratio == 0):
                    pass
                else :
                  for i in range(state_dict[key].shape[0]):
                    filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                    l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                    imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    

              elif key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'):
                print(key)
                l1_norm = state_dict[key].view(-1).abs()
                _, k_idx = torch.sort(l1_norm)
                keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
                layer_var = torch.var(l1_norm) * args.scale
                l1_norm[keep_kernel] = 1000
                kernel_score = l1_norm * (1 - beta * layer_var)
                mask = torch.tensor(kernel_score.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                rv_idx = int(state_dict[key].shape[1]*ratio)
                if ratio==0:
                  pass
                else:
                  for i in range(state_dict[key].shape[0]):
                    var = state_dict[key][i].abs().view(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            
            elif key.endswith('fc.weight'):
              chan = state_dict[key].shape[0]
              val = state_dict[key].abs().view(-1)
              f, _ = torch.sort(val)
              threshold_idx = int(val.shape[0] * args.ratio)
              threshold = f[threshold_idx].to(device)
              for i in range(chan):
                tmp = state_dict[key][i].abs().view(-1)
                mask = torch.tensor(tmp.gt(threshold))
                mask = ~mask
                mask_idx = mask.nonzero().view(-1)
                state_dict[key][i,mask_idx] = torch.zeros(1).type(torch.cuda.FloatTensor)
                  

          torch.save(state_dict, os.path.join(args.save, 'Vector_whole_ResNet50_'+str(args.ratio*100)+'%_a_'+str(args.alpha)+'_b_'+str(args.beta)+'_s_'+str(int(args.scale))+'_pruned.pth'))

        #3by3 L1 Norm + Vector
        #1by1 L1 Norm
        elif args.mode==2:
          #3by3 Conv total
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              for i in range(state_dict[key].shape[0]):
                size = state_dict[key].shape[1]
                filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                cv[index:(index+size)] = imp_score
                index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          #1by1 Conv L1 Norm + Layer Variance
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              size = state_dict[key].view(-1).shape[0]
              l1_norm = state_dict[key].view(-1).abs()
              _, k_idx = torch.sort(l1_norm)
              keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
              l1_norm[keep_kernel] = 1000
              norm[index2:(index2+size)] = l1_norm
              index2 += size
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('weight'):
              if key.endswith('conv2.weight'):
                print(key)
                zero_sum = 0
                for i in range(state_dict[key].shape[0]):
                  filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                  stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                  stand_vector[:] = filter_vector
                  cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                  l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                  alpha = args.alpha
                  imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                  mask = torch.tensor(imp_score.gt(thre))
                  mask_idx = mask.nonzero()
                  zero_sum += torch.sum(mask)
                ratio = 1 - zero_sum/float(state_dict[key].shape[0] * state_dict[key].shape[1])            
                rv_idx = int(state_dict[key].shape[1]*ratio)                
                if (ratio == 0):
                    pass
                else :
                  for i in range(state_dict[key].shape[0]):
                    #filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                    #stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                    #stand_vector[:] = filter_vector
                    #cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                    l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                    #imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(l1_norm)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    

              elif key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'):
                print(key)
                l1_norm = state_dict[key].view(-1).abs()
                _, k_idx = torch.sort(l1_norm)
                keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
                l1_norm[keep_kernel] = 1000
                kernel_score = l1_norm
                mask = torch.tensor(kernel_score.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                rv_idx = int(state_dict[key].shape[1]*ratio)
                if ratio==0:
                  pass
                else:
                  for i in range(state_dict[key].shape[0]):
                    var = state_dict[key][i].abs().view(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            
            elif key.endswith('fc.weight'):
              chan = state_dict[key].shape[0]
              val = state_dict[key].abs().view(-1)
              f, _ = torch.sort(val)
              threshold_idx = int(val.shape[0] * args.ratio)
              threshold = f[threshold_idx].to(device)
              for i in range(chan):
                tmp = state_dict[key][i].abs().view(-1)
                mask = torch.tensor(tmp.gt(threshold))
                mask = ~mask
                mask_idx = mask.nonzero().view(-1)
                state_dict[key][i,mask_idx] = torch.zeros(1).type(torch.cuda.FloatTensor)
                  

          torch.save(state_dict, os.path.join(args.save, 'Vector_wo_layer_ResNet50_'+str(args.ratio*100)+'%_a_'+str(args.alpha)+'_pruned.pth'))
        
         
          
        #3by3 L1 Norm + Vector + Prev BN scaling factor
        #1by1 L1 Norm + Prev BN scaling factor
        elif args.mode==3:
          #collect name
          name = []
          for key in list(state_dict.keys()):
            if state_dict[key].dim()==4 or ((key.find('bn')>0) and key.endswith('weight')) or key.endswith('downsample.0.weight') or key.endswith('downsample.1.weight'):
              name.append(key)
        
          #3by3 Conv total
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              idx = name.index(key) - 1
              bn_scale = state_dict[name[idx]].abs()
              prev_val = (1 - ((bn_scale.max() - bn_scale)/(bn_scale.max() + bn_scale.min())) * args.beta)
              for i in range(state_dict[key].shape[0]):
                size = state_dict[key].shape[1]
                filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * prev_val
                cv[index:(index+size)] = imp_score
                index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          #1by1 Conv L1 Norm
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              if key.endswith('downsample.0.weight'):
                idx = name.index(key) - 7
              else:
                idx = name.index(key) - 1
              bn_scale = state_dict[name[idx]].abs() 
              if key.endswith('.1.conv1.weight'):
                bn_scale += state_dict[name[idx-2]].abs()   
              prev_val = (1 - ((bn_scale.max() - bn_scale)/(bn_scale.max() + bn_scale.min())) * args.beta).reshape(state_dict[key].shape[1], 1, 1)
              size = state_dict[key].view(-1).shape[0]
              l1_norm = state_dict[key].abs() * prev_val
              l1_norm = l1_norm.view(-1)
              _, k_idx = torch.sort(l1_norm)
              keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
              l1_norm[keep_kernel] = 1000
              norm[index2:(index2+size)] = l1_norm
              index2 += size
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('weight'):
              if key.endswith('conv2.weight'):
                print(key)
                zero_sum = 0
                idx = name.index(key) - 1
                bn_scale = state_dict[name[idx]].abs()
                prev_val = (1 - ((bn_scale.max() - bn_scale)/(bn_scale.max() + bn_scale.min())) * args.beta)
                for i in range(state_dict[key].shape[0]):
                  filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                  stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                  stand_vector[:] = filter_vector
                  cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                  l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                  alpha = args.alpha
                  imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * prev_val
                  mask = torch.tensor(imp_score.gt(thre))
                  mask_idx = mask.nonzero()
                  zero_sum += torch.sum(mask)
                ratio = 1 - zero_sum/float(state_dict[key].shape[0] * state_dict[key].shape[1])            
                rv_idx = int(state_dict[key].shape[1]*ratio)                
                if (ratio == 0):
                    pass
                else :
                  for i in range(state_dict[key].shape[0]):
                    filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                    l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                    imp_score = l1_norm * (alpha + cos_val.abs() * (1 - alpha)) * prev_val
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    

              elif key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'):
                print(key)
                if key.endswith('downsample.0.weight'):
                  idx = name.index(key) - 7
                else:
                  idx = name.index(key) - 1
                bn_scale = state_dict[name[idx]].abs()
                if key.endswith('.1.conv1.weight'):
                  bn_scale += state_dict[name[idx-2]].abs()
                prev_val = (1 - ((bn_scale.max() - bn_scale)/(bn_scale.max() + bn_scale.min())) * args.beta).reshape(state_dict[key].shape[1], 1, 1)
                l1_norm = state_dict[key].abs() * prev_val
                l1_norm = l1_norm.view(-1)
                _, k_idx = torch.sort(l1_norm)
                keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
                l1_norm[keep_kernel] = 1000
                kernel_score = l1_norm
                mask = torch.tensor(kernel_score.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                rv_idx = int(state_dict[key].shape[1]*ratio)
                if ratio==0:
                  pass
                else:
                  for i in range(state_dict[key].shape[0]):
                    var = state_dict[key][i].abs().view(-1) * prev_val.reshape(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            
            elif key.endswith('fc.weight'):
              chan = state_dict[key].shape[0]
              val = state_dict[key].abs().view(-1)
              f, _ = torch.sort(val)
              threshold_idx = int(val.shape[0] * args.ratio)
              threshold = f[threshold_idx].to(device)
              for i in range(chan):
                tmp = state_dict[key][i].abs().view(-1)
                mask = torch.tensor(tmp.gt(threshold))
                mask = ~mask
                mask_idx = mask.nonzero().view(-1)
                state_dict[key][i,mask_idx] = torch.zeros(1).type(torch.cuda.FloatTensor)
                  

          torch.save(state_dict, os.path.join(args.save, 'Vector_BN_ResNet50_'+str(args.ratio*100)+'%_a_'+str(args.alpha)+'_b_'+str(args.beta)+'_pruned.pth'))  
        
        #3by3 L1 Norm + Vector (no abs)
        elif args.mode==4:
          #3by3 Conv total
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              for i in range(state_dict[key].shape[0]):
                size = state_dict[key].shape[1]
                filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm * (alpha + cos_val * (1 - alpha))
                cv[index:(index+size)] = imp_score
                index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          #1by1 Conv L1 Norm + Layer Variance
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              size = state_dict[key].view(-1).shape[0]
              l1_norm = state_dict[key].view(-1).abs()
              _, k_idx = torch.sort(l1_norm)
              keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
              l1_norm[keep_kernel] = 1000
              norm[index2:(index2+size)] = l1_norm
              index2 += size
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('weight'):
              if key.endswith('conv2.weight'):
                print(key)
                zero_sum = 0
                for i in range(state_dict[key].shape[0]):
                  filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                  stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                  stand_vector[:] = filter_vector
                  cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                  l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                  alpha = args.alpha
                  imp_score = l1_norm * (alpha + cos_val * (1 - alpha))
                  mask = torch.tensor(imp_score.gt(thre))
                  mask_idx = mask.nonzero()
                  zero_sum += torch.sum(mask)
                ratio = 1 - zero_sum/float(state_dict[key].shape[0] * state_dict[key].shape[1])            
                rv_idx = int(state_dict[key].shape[1]*ratio)                
                if (ratio == 0):
                    pass
                else :
                  for i in range(state_dict[key].shape[0]):
                    filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                    l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                    imp_score = l1_norm * (alpha + cos_val * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    

              elif key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'):
                print(key)
                l1_norm = state_dict[key].view(-1).abs()
                _, k_idx = torch.sort(l1_norm)
                keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
                l1_norm[keep_kernel] = 1000
                kernel_score = l1_norm
                mask = torch.tensor(kernel_score.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                rv_idx = int(state_dict[key].shape[1]*ratio)
                if ratio==0:
                  pass
                else:
                  for i in range(state_dict[key].shape[0]):
                    var = state_dict[key][i].abs().view(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            
            elif key.endswith('fc.weight'):
              chan = state_dict[key].shape[0]
              val = state_dict[key].abs().view(-1)
              f, _ = torch.sort(val)
              threshold_idx = int(val.shape[0] * args.ratio)
              threshold = f[threshold_idx].to(device)
              for i in range(chan):
                tmp = state_dict[key][i].abs().view(-1)
                mask = torch.tensor(tmp.gt(threshold))
                mask = ~mask
                mask_idx = mask.nonzero().view(-1)
                state_dict[key][i,mask_idx] = torch.zeros(1).type(torch.cuda.FloatTensor)
                  

          torch.save(state_dict, os.path.join(args.save, 'L1_no_abs_layer_ResNet50_'+str(args.ratio*100)+'%_a_'+str(args.alpha)+'_pruned.pth'))
          
        #3by3 Only L2 Norm Vector
        elif args.mode==5:
          #3by3 Conv total
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              for i in range(state_dict[key].shape[0]):
                size = state_dict[key].shape[1]
                filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                stand_vector[:] = filter_vector
                cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                alpha = args.alpha
                imp_score = l1_norm.norm() * (alpha + cos_val.abs() * (1 - alpha))
                cv[index:(index+size)] = imp_score
                index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          #1by1 Conv L1 Norm + Layer Variance
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              size = state_dict[key].view(-1).shape[0]
              l1_norm = state_dict[key].view(-1).abs()
              _, k_idx = torch.sort(l1_norm)
              keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
              l1_norm[keep_kernel] = 1000
              norm[index2:(index2+size)] = l1_norm
              index2 += size
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('weight'):
              if key.endswith('conv2.weight'):
                print(key)
                zero_sum = 0
                for i in range(state_dict[key].shape[0]):
                  filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                  stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                  stand_vector[:] = filter_vector
                  cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                  l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                  alpha = args.alpha
                  imp_score = l1_norm.norm() * (alpha + cos_val.abs() * (1 - alpha))
                  mask = torch.tensor(imp_score.gt(thre))
                  mask_idx = mask.nonzero()
                  zero_sum += torch.sum(mask)
                ratio = 1 - zero_sum/float(state_dict[key].shape[0] * state_dict[key].shape[1])            
                rv_idx = int(state_dict[key].shape[1]*ratio)                
                if (ratio == 0):
                    pass
                else :
                  for i in range(state_dict[key].shape[0]):
                    filter_vector = state_dict[key][i].reshape(state_dict[key].shape[1], -1).sum(dim=0)
                    stand_vector = torch.zeros(state_dict[key].shape[1], state_dict[key].shape[2] * state_dict[key].shape[3]).type(torch.cuda.FloatTensor)
                    stand_vector[:] = filter_vector
                    cos_val = cos(stand_vector, state_dict[key][i].reshape(state_dict[key].shape[1], -1))
                    l1_norm = state_dict[key][i].reshape(state_dict[key].shape[1], -1).abs().sum(dim=1)
                    imp_score = l1_norm.norm() * (alpha + cos_val.abs() * (1 - alpha))
                    _, j = torch.sort(imp_score)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
    

              elif key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight'):
                print(key)
                l1_norm = state_dict[key].view(-1).abs()
                _, k_idx = torch.sort(l1_norm)
                keep_kernel = k_idx[-int(l1_norm.shape[0]*args.per):]
                l1_norm[keep_kernel] = 1000
                kernel_score = l1_norm
                mask = torch.tensor(kernel_score.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                rv_idx = int(state_dict[key].shape[1]*ratio)
                if ratio==0:
                  pass
                else:
                  for i in range(state_dict[key].shape[0]):
                    var = state_dict[key][i].abs().view(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            
            elif key.endswith('fc.weight'):
              chan = state_dict[key].shape[0]
              val = state_dict[key].abs().view(-1)
              f, _ = torch.sort(val)
              threshold_idx = int(val.shape[0] * args.ratio)
              threshold = f[threshold_idx].to(device)
              for i in range(chan):
                tmp = state_dict[key][i].abs().view(-1)
                mask = torch.tensor(tmp.gt(threshold))
                mask = ~mask
                mask_idx = mask.nonzero().view(-1)
                state_dict[key][i,mask_idx] = torch.zeros(1).type(torch.cuda.FloatTensor)
                  

          torch.save(state_dict, os.path.join(args.save, 'L2_only_vector_layer_ResNet50_'+str(args.ratio*100)+'%_a_'+str(args.alpha)+'_pruned.pth'))
        

        else:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and key.endswith('conv2.weight'):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].to(device)
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight')):
              total2 += state_dict[key].view(-1).shape[0]
          norm = torch.zeros(total2)
          index2 = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight')):
              size2 = state_dict[key].view(-1).shape[0]
              norm[index2:(index2+size2)] = state_dict[key].view(-1).abs()
              index2 += size2
          z, _ = torch.sort(norm)
          thres_index = int(total2 * args.ratio)
          thres = z[thres_index].to(device)
          print('\nthreshold of 1by1 :', thres,'\n')
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv2.weight'):
                weight_copy = state_dict[key].view(-1,9).abs().sum(dim=1).clone()
                mask = torch.tensor(weight_copy.gt(thre))
                mask_idx = mask.nonzero()
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1,9).shape[0])
                
                chan = state_dict[key].shape[0]
                rv_idx = int(state_dict[key].shape[1]*ratio)
                #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
              elif key.endswith('conv1.weight') or key.endswith('conv3.weight'):
                weight_copy = state_dict[key].view(-1).abs().clone()
                mask = torch.tensor(weight_copy.gt(thres))
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1).shape[0])
                chan = state_dict[key].shape[0]
                rv_idx = int(state_dict[key].shape[1]*ratio)
                #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
              
              if (ratio == 0):
                pass
              else :
                #print(key)
                #print('weight_mask :', weight_mask.shape)
                chan = state_dict[key].shape[0]
                if key.endswith('conv2.weight'):
                  for i in range(chan):
                    var = state_dict[key][i].abs().view(state_dict[key].shape[1],-1).sum(dim=1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                
                elif key.endswith('conv1.weight') or key.endswith('conv3.weight'):
                  for i in range(chan):
                    var = state_dict[key][i].abs().view(-1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)

          torch.save(state_dict, os.path.join(args.save, 'Mask_ResNet50_'+str(args.ratio*100)+'%_pruned.pth')) 


pruning = Prune()
pruning.load_weights(path)
print('Pruning Proccess finished')

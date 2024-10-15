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

parser.add_argument('--path', default='weights/', 
        help='Directory for load weight.')
parser.add_argument('--ratio', default=0.1, type=float,
        help='Ratio for pruning weights.')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')
parser.add_argument('--mode', default=3, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')

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
        name = []
        model_similarity = {}
        for key in list(state_dict.keys()):
          if key.startswith('module.layer'):
            name.append(key)


        #LSE Kernel Pruning
        if args.mode==1:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].cuda()
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                weight_copy = state_dict[key].view(-1,9).abs().sum(dim=1).clone()
                mask = torch.tensor(weight_copy.gt(thre))
                mask_idx = mask.nonzero()
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1,9).shape[0])
                
                chan = state_dict[key].shape[0]
                rv_idx = int(state_dict[key].shape[1]*ratio)
                #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
              if (ratio == 0):
                pass
              else :
                #print(key)
                #print('weight_mask :', weight_mask.shape)
                chan = state_dict[key].shape[0]
                if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                  for i in range(chan):
                    var = state_dict[key][i].abs().view(state_dict[key].shape[1],-1).sum(dim=1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)

            elif key.endswith('linear.weight'):
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
                  
          torch.save(state_dict, os.path.join(args.save, 'LSE_ResNet56_'+str(args.ratio*100)+'%_pruned.pth')) 
          
          
        #Adaptive Kernel Pruning without FC
        elif args.mode==2:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].cuda()
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                #weight_mask = torch.ones(state_dict[key].shape[0],state_dict[key].shape[1])
                chan = state_dict[key].shape[0]
                weight_tmp = state_dict[key]
                for i in range(chan):
                  tmp = state_dict[key][i].view(-1,9).abs().sum(dim=1)
                  max_idx = torch.argmax(tmp)
                  mask = torch.tensor(tmp.gt(thre))
                  mask = ~mask
                  #weight_mask[i] = mask
                  mask_idx = mask.nonzero().view(-1)
                  if mask_idx.shape[0] == state_dict[key].shape[1]:
                      state_dict[key][i,mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                      state_dict[key][i,max_idx,:,:] = weight_tmp[i,max_idx,:,:]
                  else:
                      state_dict[key][i,mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                #mask_name = key[:-6]
                #mask_name = mask_name + 'mask'
                #weight_mask = weight_mask.type(torch.cuda.BoolTensor)
                #state_dict[mask_name] = weight_mask
                  #print(key, ':', rv_idx)
                  #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
          
          torch.save(state_dict, os.path.join(args.save, 'Adaptive_ResNet56_'+str(args.ratio*100)+'%_pruned.pth'))
          
        #LSE Kernel Pruning without FC
        elif args.mode==3:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].cuda()
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                weight_copy = state_dict[key].view(-1,9).abs().sum(dim=1).clone()
                mask = torch.tensor(weight_copy.gt(thre))
                mask_idx = mask.nonzero()
                ratio = 1 - torch.sum(mask)/float(state_dict[key].view(-1,9).shape[0])
                
                chan = state_dict[key].shape[0]
                rv_idx = int(state_dict[key].shape[1]*ratio)
                #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
              if (ratio == 0):
                pass
              else :
                #print(key)
                #print('weight_mask :', weight_mask.shape)
                chan = state_dict[key].shape[0]
                if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                  for i in range(chan):
                    var = state_dict[key][i].abs().view(state_dict[key].shape[1],-1).sum(dim=1)
                    val, j = torch.sort(var)
                    val_mask = j[:rv_idx]
                    val_mask_idx, _ = torch.sort(val_mask)
                    state_dict[key][i,val_mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor) 
                  
          torch.save(state_dict, os.path.join(args.save, 'LSE_ResNet56_wo_'+str(args.ratio*100)+'%_pruned.pth')) 
          
        
        #Adaptive L2 Kernel Pruning without FC
        elif args.mode==4:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].cuda()
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                #weight_mask = torch.ones(state_dict[key].shape[0],state_dict[key].shape[1])
                chan = state_dict[key].shape[0]
                for i in range(chan):
                  tmp = state_dict[key][i].view(-1,9).abs().sum(dim=1)
                  mask = torch.tensor(tmp.gt(thre))
                  mask = ~mask
                  #weight_mask[i] = mask
                  mask_idx = mask.nonzero().view(-1)
                  state_dict[key][i,mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                #mask_name = key[:-6]
                #mask_name = mask_name + 'mask'
                #weight_mask = weight_mask.type(torch.cuda.BoolTensor)
                #state_dict[mask_name] = weight_mask
                  #print(key, ':', rv_idx)
                  #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
          
          torch.save(state_dict, os.path.join(args.save, 'Adaptive_ResNet56_'+str(args.ratio*100)+'%_pruned.pth'))  
          
          
          
        #Adaptive Kernel Pruning
        else:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              total += state_dict[key].view(-1,9).shape[0]
          cv = torch.zeros(total)
          index = 0
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv2.weight')):
              size = state_dict[key].view(-1,9).shape[0]
              cv[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
              index += size
          y, _ = torch.sort(cv)
          thre_index = int(total * args.ratio)
          thre = y[thre_index].cuda()
          print('\nthre ', thre)
          
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv')>0) and (key.find('weight')>0):
              idx = name.index(key)
              if key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                #weight_mask = torch.ones(state_dict[key].shape[0],state_dict[key].shape[1])
                chan = state_dict[key].shape[0]
                for i in range(chan):
                  tmp = state_dict[key][i].view(-1,9).abs().sum(dim=1)
                  mask = torch.tensor(tmp.gt(thre))
                  mask = ~mask
                  #weight_mask[i] = mask
                  mask_idx = mask.nonzero().view(-1)
                  state_dict[key][i,mask_idx,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
                #mask_name = key[:-6]
                #mask_name = mask_name + 'mask'
                #weight_mask = weight_mask.type(torch.cuda.BoolTensor)
                #state_dict[mask_name] = weight_mask
                  #print(key, ':', rv_idx)
                  #weight_mask = torch.zeros(chan, state_dict[key].shape[1] - rv_idx).type(torch.LongTensor)
            
            
            elif key.endswith('linear.weight'):
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
          
          torch.save(state_dict, os.path.join(args.save, 'Adaptive_ResNet56_'+str(args.ratio*100)+'%_pruned.pth')) 



pruning = Prune()
pruning.load_weights(path)
print('Pruning Proccess finished')

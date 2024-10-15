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
parser.add_argument('--save', default='structured/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')
parser.add_argument('--mode', default=1, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')


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

class Prune(nn.Module):
    def __init__(self):
        print('Kernel Pruning Process Start!')
    def load_weights(self, path):
        '''Loads weights from a compressed save file.'''
        state_dict = torch.load(path)
         
        #ResNet 50 Unstructured Pruning to Structured Pruning 
        if args.mode==1:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0) or (key.find('downsample.0.')>0)) and (key.find('weight')>0):
              print(key)     
              mask = (state_dict[key][0].view(-1) == 0)
              mask = ~mask
              mask = mask.nonzero().view(-1)
              weight_mask = torch.zeros(state_dict[key].shape[0], mask.shape[0]).type(torch.LongTensor)
              weight_mask[0] = mask
              temp = state_dict[key][0,mask,:,:].unsqueeze(0)
              for i in range(1, state_dict[key].shape[0]):
                mask = (state_dict[key][i].view(-1) == 0)
                mask = ~mask
                mask = mask.nonzero().view(-1)
                weight_mask[i] = mask
                temp = torch.cat([temp, state_dict[key][i,weight_mask[i],:,:].unsqueeze(0)], dim=0)
              state_dict[key] = temp
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              weight_mask = weight_mask.type(torch.IntTensor)
              state_dict[mask_name] = weight_mask         
            
            elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
              print(key)
              mask = (state_dict[key][0].view(-1,9).sum(dim=1) == 0)
              mask = ~mask
              mask = mask.nonzero().view(-1)
              weight_mask = torch.zeros(state_dict[key].shape[0], mask.shape[0]).type(torch.LongTensor)
              weight_mask[0] = mask
              temp = state_dict[key][0,mask,:,:].unsqueeze(0)
              for i in range(1, state_dict[key].shape[0]):
                mask = (state_dict[key][i].view(-1,9).sum(dim=1) == 0)
                mask = ~mask
                mask = mask.nonzero().view(-1)
                weight_mask[i] = mask
                temp = torch.cat([temp, state_dict[key][i,weight_mask[i],:,:].unsqueeze(0)], dim=0)
              state_dict[key] = temp
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              weight_mask = weight_mask.type(torch.IntTensor)
              state_dict[mask_name] = weight_mask
                    
          torch.save(state_dict, os.path.join(args.save, 'ResNet50_79.5%.pth'))
          
        else:
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0)) and (key.find('weight')>0):
              print(key)     
              mask = (state_dict[key][0].view(-1) == 0)
              mask = ~mask
              mask = mask.nonzero().view(-1)
              weight_mask = torch.zeros(state_dict[key].shape[0], mask.shape[0]).type(torch.LongTensor)
              weight_mask[0] = mask
              temp = state_dict[key][0,mask,:,:].unsqueeze(0)
              for i in range(1, state_dict[key].shape[0]):
                mask = (state_dict[key][i].view(-1) == 0)
                mask = ~mask
                mask = mask.nonzero().view(-1)
                weight_mask[i] = mask
                temp = torch.cat([temp, state_dict[key][i,weight_mask[i],:,:].unsqueeze(0)], dim=0)
              state_dict[key] = temp
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              weight_mask = weight_mask.type(torch.IntTensor)
              state_dict[mask_name] = weight_mask         
            
            elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
              print(key)
              mask = (state_dict[key][0].view(-1,9).sum(dim=1) == 0)
              mask = ~mask
              mask = mask.nonzero().view(-1)
              weight_mask = torch.zeros(state_dict[key].shape[0], mask.shape[0]).type(torch.LongTensor)
              weight_mask[0] = mask
              temp = state_dict[key][0,mask,:,:].unsqueeze(0)
              for i in range(1, state_dict[key].shape[0]):
                mask = (state_dict[key][i].view(-1,9).sum(dim=1) == 0)
                mask = ~mask
                mask = mask.nonzero().view(-1)
                weight_mask[i] = mask
                temp = torch.cat([temp, state_dict[key][i,weight_mask[i],:,:].unsqueeze(0)], dim=0)
              state_dict[key] = temp
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              weight_mask = weight_mask.type(torch.IntTensor)
              state_dict[mask_name] = weight_mask
            
            elif key.startswith('module.fc.weight'):
              mask = (state_dict[key][0].view(-1) == 0)
              mask = ~mask
              mask = mask.nonzero().view(-1)
              weight_mask = torch.zeros(state_dict[key].shape[0], mask.shape[0]).type(torch.LongTensor)
              weight_mask[0] = mask
              temp = state_dict[key][0,mask].unsqueeze(0)
              for i in range(1, state_dict[key].shape[0]):
                mask = (state_dict[key][i].view(-1) == 0)
                mask = ~mask
                mask = mask.nonzero().view(-1)
                weight_mask[i] = mask
                temp = torch.cat([temp, state_dict[key][i,weight_mask[i]].unsqueeze(0)], dim=0)
              state_dict[key] = temp
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              weight_mask = weight_mask.type(torch.IntTensor)
              state_dict[mask_name] = weight_mask
                    
          torch.save(state_dict, os.path.join(args.save, 'ResNet50_fc_79.5%.pth'))



pruning = Prune()
pruning.load_weights(path)
print('Pruning Proccess finished')

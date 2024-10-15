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
          name = []
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight') or key.endswith('conv2.weight')):
              name.append(key)
              print(key)
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
              filter_num = state_dict[key].shape[0]
              kernel_num = state_dict[key].shape[1]
              unit_bit = torch.ceil(torch.tensor(kernel_num/8)).type(torch.cuda.IntTensor)
              temp_mask = torch.zeros(filter_num, unit_bit).type(torch.cuda.ByteTensor)
              for i in range(filter_num):
                mask = (state_dict[key][i].view(-1) != 0)
                for j in range(unit_bit):
                  b_val = '0b'
                  for b in range(8):
                    b_val += str(int(mask[8 * j + b]))
                  temp_mask[i,j] = torch.tensor(int(b_val, 2))
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              #weight_mask = weight_mask.type(torch.cuda.ByteTensor)
              state_dict[mask_name] = temp_mask
              #print(key, '\n', temp_mask.shape)  
            
            elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0): 
              filter_num = state_dict[key].shape[0]
              kernel_num = state_dict[key].shape[1]
              unit_bit = torch.ceil(torch.tensor(kernel_num/8)).type(torch.cuda.IntTensor)
              temp_mask = torch.zeros(filter_num, unit_bit).type(torch.cuda.ByteTensor)
              for i in range(filter_num):
                mask = (state_dict[key][i].view(-1,9).sum(dim=1) != 0)
                for j in range(unit_bit):
                  b_val = '0b'
                  for b in range(8):
                    b_val += str(int(mask[8 * j + b]))
                  temp_mask[i,j] = torch.tensor(int(b_val, 2))
              mask_name = key[:-6]
              mask_name = mask_name + 'mask'
              #weight_mask = weight_mask.type(torch.cuda.ByteTensor)
              state_dict[mask_name] = temp_mask
              #print(key, '\n', temp_mask.shape)  
        
        
          for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
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
              state_dict[mask_name] = weight_mask
                    
          torch.save(state_dict, os.path.join(args.save, 'Decoded_ResNet50_79.5%.pth'))



pruning = Prune()
pruning.load_weights(path)
print('Pruning Proccess finished')

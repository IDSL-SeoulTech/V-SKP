import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

import argparse
import datetime
import os
from torch.autograd import Variable
from torchvision import datasets, transforms
import time

parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='init_weight/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')

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
        state_dict = torch.load(path)
        print(type(state_dict))
        for key in list(state_dict.keys()):
          print(key, state_dict[key].shape)
          
          if key.startswith('module.layer') and key.endswith('conv2.weight'):
            f_num = state_dict[key].shape[0]
            val = state_dict[key].clone()
            new_weight = init.xavier_uniform_(val)
            for f in range(f_num):
              weight_mask = (state_dict[key][f].view(-1,9).sum(dim=1) == 0)
              mask = weight_mask.nonzero().view(-1)
              new_weight[f,mask,:,:] = torch.zeros(3,3).type(torch.cuda.FloatTensor)
            state_dict[key] = new_weight

          elif key.startswith('module.layer') and (key.endswith('conv1.weight') or key.endswith('conv3.weight') or key.endswith('downsample.0.weight')):
            f_num = state_dict[key].shape[0]
            val = state_dict[key].clone()
            new_weight = init.xavier_uniform_(val)
            for f in range(f_num):
              weight_mask = (state_dict[key][f].view(-1) == 0)
              mask = weight_mask.nonzero().view(-1)
              new_weight[f,mask,:,:] = torch.zeros(1).type(torch.cuda.FloatTensor)
            state_dict[key] = new_weight
            
          elif key.startswith('module.fc.weight'):
            f_num = state_dict[key].shape[0]
            val = state_dict[key].clone()
            new_weight = init.xavier_uniform_(val)
            for f in range(f_num):
              weight_mask = (state_dict[key][f].view(-1) == 0)
              mask = weight_mask.nonzero().view(-1)
              new_weight[f,mask] = torch.zeros(1).type(torch.cuda.FloatTensor)
            state_dict[key] = new_weight
          
          elif key.startswith('module.conv1.weight'):
            val = state_dict[key].clone()
            state_dict[key] = init.xavier_uniform_(val)
          elif key.startswith('module.bn1.running_mean'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.zeros(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.bn1.running_var'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.ones(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.bn1.weight'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.ones(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.bn1.bias'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.zeros(f_num).type(torch.cuda.FloatTensor)

          elif key.startswith('module.layer') and ((key.find('bn')>0) or (key.find('downsample.1.')>0)) and key.endswith('running_mean'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.zeros(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.layer') and ((key.find('bn')>0) or (key.find('downsample.1.')>0)) and key.endswith('running_var'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.ones(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.layer') and ((key.find('bn')>0) or (key.find('downsample.1.')>0)) and key.endswith('weight'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.ones(f_num).type(torch.cuda.FloatTensor)
          elif key.startswith('module.layer') and ((key.find('bn')>0) or (key.find('downsample.1.')>0)) and key.endswith('bias'):
            f_num = state_dict[key].shape[0]
            state_dict[key] = torch.zeros(f_num).type(torch.cuda.FloatTensor)
          
        torch.save(state_dict, os.path.join(args.save, 'Init_ResNet50.pth')) 


pruning = Prune()
pruning.load_weights(path)
print('Pruning Proccess finished')

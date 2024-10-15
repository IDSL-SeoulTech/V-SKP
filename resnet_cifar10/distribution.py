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
import csv
import pandas as pd

parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='csv/', type=str, metavar='PATH', help='path to save pruned model (default : pruned_weights/')

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


class Check(nn.Module):
    def __init__(self):
        super().__init__()
        
    def load_weights(self, origin_path):
        '''Loads weights from a compressed save file.'''
        total=0
        origin_weight = torch.load(origin_path)
        self.origin = {}
        
        for key in list(origin_weight.keys()):
            if key.startswith('module.layer') and key.endswith('conv1.weight') or key.endswith('conv2.weight'):
                weight = origin_weight[key].clone()
                kernel_norm = weight.reshape(-1, weight.shape[2] * weight.shape[3]).abs().sum(dim=1)
                kernel_aligned, _ = kernel_norm.sort()
                idx = kernel_aligned.shape[0] - 1
                sorted_norm = torch.zeros(11)
                for i in range(11):
                    sorted_norm[i] = kernel_aligned[int(idx*0.1*i)]
                self.origin[key[12:21]] = sorted_norm.cpu().numpy()
        
            

    def make_csv(self):
        weight_layer = self.origin    
        owl = pd.DataFrame(weight_layer)     
        owl.to_csv(args.save + 'weight_distribution.csv')

        print('CSV')

checking = Check()
checking.load_weights(path)
checking.make_csv()
print('Finished')



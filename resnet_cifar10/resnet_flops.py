import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import time
from datetime import timedelta

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(p=0.05)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x[0])
        total = self.conv1.weight.view(-1).shape[0]
        actual = (self.conv1.weight.view(-1) == 0).sum()
        pruned_flops = actual * out.shape[2] * out.shape[3] + x[1]
        total_flops = total * out.shape[2] * out.shape[3] + x[2]
        
        out = self.bn1(out)
        total = self.bn1.weight.shape[0]
        total_flops += total * out.shape[2] * out.shape[3]
        
        out = F.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        actual = (self.conv2.weight.view(-1) == 0).sum()
        total = self.conv2.weight.view(-1).shape[0]
        pruned_flops += actual * out.shape[2] * out.shape[3]
        total_flops += total * out.shape[2] * out.shape[3]
        
        out = self.bn2(out)
        total = self.bn2.weight.shape[0]
        total_flops += total * out.shape[2] * out.shape[3]
        
        out = self.drop(out)        
        out += self.shortcut(x[0])
        out = F.relu(out)
        
        return [out, pruned_flops, total_flops]


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #start = time.process_time()
        out = self.conv1(x)
        total = self.conv1.weight.view(-1).shape[0]
        pruned_flops = 0
        total_flops = total * out.shape[2] * out.shape[3]
        print('\nconv1 pruned flops :', pruned_flops)
        print('conv1 total flops :', total_flops)
        
        out = self.bn1(out)
        total = self.bn1.weight.shape[0]
        total_flops += total * out.shape[2] * out.shape[3]
        out = F.relu(out)
        
        out = self.layer1([out, pruned_flops, total_flops])
        print('\nlayer1 pruned flops:', int(out[1]))
        print('\nlayer1 total flops:', out[2])
        
        out = self.layer2(out)
        print('\nlayer2 pruned flops:', int(out[1]))
        print('\nlayer2 total flops:', out[2])
        
        #print('layer2 :', out.shape)
        out = self.layer3(out)
        pruned_flops = int(out[1])
        total_flops = out[2]
        print('\nlayer3 pruned flops:', pruned_flops)
        print('\nlayer3 total flops:', total_flops)
        #print('layer3 :', out.shape)
        
        out = F.avg_pool2d(out[0], out[0].size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        total = self.linear.weight.view(-1).shape[0]
        actual = (self.linear.weight.view(-1) == 0).sum()
        pruned_flops += actual
        total_flops += total
        print('\nTotal Pruned FLOPs :', int(pruned_flops))
        print('Total Original FLOPs :', total_flops)
        print()
        print('FLOPs Drop Rate :', '%.2f' %(((pruned_flops/total_flops))*100), '%')
        print()
        print()
        
        exit()
        
        #end = time.process_time()
        #print('CPU Latency :', end-start)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
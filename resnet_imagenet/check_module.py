import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from models import *
from data_loader import data_loader
from helper import AverageMeter, accuracy, adjust_learning_rate

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre_trained model')
parser.add_argument('--path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


args = parser.parse_args()

    # create model
if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
else:
    print("=> creating model '{}'".format(args.arch))

if args.arch == 'alexnet':
    model = alexnet(pretrained=args.pretrained)
elif args.arch == 'squeezenet1_0':
    model = squeezenet1_0(pretrained=args.pretrained)
elif args.arch == 'squeezenet1_1':
    model = squeezenet1_1(pretrained=args.pretrained)
elif args.arch == 'densenet121':
    model = densenet121(pretrained=args.pretrained)
elif args.arch == 'densenet169':
    model = densenet169(pretrained=args.pretrained)
elif args.arch == 'densenet201':
    model = densenet201(pretrained=args.pretrained)
elif args.arch == 'densenet161':
    model = densenet161(pretrained=args.pretrained)
elif args.arch == 'vgg11':
    model = vgg11(pretrained=args.pretrained)
elif args.arch == 'vgg13':
    model = vgg13(pretrained=args.pretrained)
elif args.arch == 'vgg16':
    model = vgg16(pretrained=args.pretrained)
elif args.arch == 'vgg19':
    model = vgg19(pretrained=args.pretrained)
elif args.arch == 'vgg11_bn':
    model = vgg11_bn(pretrained=args.pretrained)
elif args.arch == 'vgg13_bn':
    model = vgg13_bn(pretrained=args.pretrained)
elif args.arch == 'vgg16_bn':
    model = vgg16_bn(pretrained=args.pretrained)
elif args.arch == 'vgg19_bn':
    model = vgg19_bn(pretrained=args.pretrained)
elif args.arch == 'resnet18':
    model = resnet18(pretrained=args.pretrained)
elif args.arch == 'resnet34':
    model = resnet34(pretrained=args.pretrained)
elif args.arch == 'resnet50':
    model = resnet50(pretrained=args.pretrained)
elif args.arch == 'resnet101':
    model = resnet101(pretrained=args.pretrained)
elif args.arch == 'resnet152':
    model = resnet152(pretrained=args.pretrained)
else:
    raise NotImplementedError

    # use cuda
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if __name__ == "__main__":
    state_dict = torch.load(args.path)
    model.load_state_dict(state_dict)
    for m in model.modules():
        print(m)
        
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import tqdm
import time
import argparse

from models import *
from data_loader import data_loader
from helper import AverageMeter, accuracy, adjust_learning_rate

def get_args():
    parser = argparse.ArgumentParser(description="ImageNet Evaluation")
    parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/data/imagenet/imagenet_val/')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-m', '--pin_memory', dest='pin_memory', action='store_true', help='use pin memory')
    parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='use pre_trained model')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--path", default=None, help="Resume from pre-train model")
    parser.add_argument("--net", default="resnet50", help="Which model")
    parser.add_argument("--max_batch", default=None, type=int)
    return parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Prepare Date

args = get_args()

_, test_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)



# create model
if args.net == 'alexnet':
    model = alexnet(pretrained=args.pretrained)
elif args.net == 'squeezenet1_0':
    model = squeezenet1_0(pretrained=args.pretrained)
elif args.net == 'squeezenet1_1':
    model = squeezenet1_1(pretrained=args.pretrained)
elif args.net == 'densenet121':
    model = densenet121(pretrained=args.pretrained)
elif args.net == 'densenet169':
    model = densenet169(pretrained=args.pretrained)
elif args.net == 'densenet201':
    model = densenet201(pretrained=args.pretrained)
elif args.net == 'densenet161':
    model = densenet161(pretrained=args.pretrained)
elif args.net == 'vgg11':
    model = vgg11(pretrained=args.pretrained)
elif args.net == 'vgg13':
    model = vgg13(pretrained=args.pretrained)
elif args.net == 'vgg16':
    model = vgg16(pretrained=args.pretrained)
elif args.net == 'vgg19':
    model = vgg19(pretrained=args.pretrained)
elif args.net == 'vgg11_bn':
    model = vgg11_bn(pretrained=args.pretrained)
elif args.net == 'vgg13_bn':
    model = vgg13_bn(pretrained=args.pretrained)
elif args.net == 'vgg16_bn':
    model = vgg16_bn(pretrained=args.pretrained)
elif args.net == 'vgg19_bn':
    model = vgg19_bn(pretrained=args.pretrained)
elif args.net == 'resnet18':
    model = resnet18(pretrained=args.pretrained)
elif args.net == 'resnet34':
    model = resnet34(pretrained=args.pretrained)
elif args.net == 'resnet50':
    model = resnet50(pretrained=args.pretrained)
elif args.net == 'resnet101':
    model = resnet101(pretrained=args.pretrained)
elif args.net == 'resnet152':
    model = resnet152(pretrained=args.pretrained)
else:
    raise NotImplementedError



net = model.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

    
def only_val(path):
    net.eval()
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    total = 0
    total_3by3 = 0
    total_1by1 = 0
    if args.net=='resnet50':
        for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
                total_3by3 += state_dict[key].view(-1,9).shape[0]
            elif key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0)) and (key.find('weight')>0):
                total_1by1 += state_dict[key].view(-1).shape[0]
        n3 = torch.zeros(total_3by3)
        n1 = torch.zeros(total_1by1)
        index = 0
        index2= 0
        for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
                size = state_dict[key].view(-1,9).shape[0]
                n3[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
                index += size
            elif key.startswith('module.layer') and ((key.find('conv1')>0) or (key.find('conv3')>0)) and (key.find('weight')>0):
                size = state_dict[key].view(-1).shape[0]
                n1[index2:(index2+size)] = state_dict[key].view(-1).abs()
                index2 += size
        y, _ = torch.sort(n3)
        z, _ = torch.sort(n1)
        print()
        for i in range(1,10):
            print('3by3 Conv L1 Norm', i*10, '% :', y[int(total_3by3 * i*0.1)]/9)
            print()
        print('\n')    
        for i in range(1,10):
            print('1by1 Conv L1 Norm', i*10, '% :', z[int(total_1by1 * i*0.1)])
            print()
        print('\n')
       
    elif args.net=='mobilenet':
        for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv1')>0) and (key.find('weight')>0):
                total_3by3 += state_dict[key].view(-1,9).shape[0]
            elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
                total_1by1 += state_dict[key].view(-1).shape[0]
        n3 = torch.zeros(total_3by3)
        n1 = torch.zeros(total_1by1)
        index = 0
        index2= 0
        
        for key in list(state_dict.keys()):
            if key.startswith('module.layer') and (key.find('conv1')>0) and (key.find('weight')>0):
                size = state_dict[key].view(-1,9).shape[0]
                n3[index:(index+size)] = state_dict[key].view(-1,9).abs().sum(dim=1)
                index += size
            elif key.startswith('module.layer') and (key.find('conv2')>0) and (key.find('weight')>0):
                size = state_dict[key].view(-1).shape[0]
                n1[index2:(index2+size)] = state_dict[key].view(-1).abs()
                index2 += size
        
        y, _ = torch.sort(n3)
        z, _ = torch.sort(n1)
        
        print()
        for i in range(1,10):
            print('3by3 Conv L1 Norm', i*10, '% :', y[int(total_3by3 * i*0.1)]/9)
            print()
        print('\n\n')    
        for i in range(1,10):
            print('1by1 Conv L1 Norm', i*10, '% :', z[int(total_1by1 * i*0.1)])
            print()
        print('\n\n')
    
    val_loss = 0
    correct = 0
    total = 0
    pbar = tqdm.tqdm(test_loader)
    for batch, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs)
        loss = criterion(output, targets)
        val_loss += loss.item()
        _, predict_maxidx = output.max(1)
        correct += predict_maxidx.eq(targets).sum().item()
        total += inputs.size(0)
        pbar.set_description()
        if args.max_batch and args.max_batch==batch:
            break
    
    print('\n',(batch+1)*args.batch_size,'Images \t Val_Acc :',100.*correct/total,'%\n')

    


if __name__ == "__main__":
    only_val(args.path)

            
            

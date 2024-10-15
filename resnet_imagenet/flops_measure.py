import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from models.resnet_flops import resnet50_flops
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
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/hdd/imagenet/')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin_memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre_trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("--save", default='weights/', help='Directory for saving checkpoint models.')
parser.add_argument('--save_interval_epoch', default=None, type=int)
parser.add_argument('--sr', dest='sr', action='store_true', help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
parser.add_argument('--fine_tuning', dest='fine_tuning', action='store_true')

best_prec1 = 0.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    global args, best_prec1
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
        model = resnet50_flops(pretrained=args.pretrained)
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
    #model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    
    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)
    
    # optionlly resume from a checkpoint
    if args.resume:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
        if args.evaluate:
            pass
        else:
            best_prec1, _ = validate(val_loader, model, criterion, args.print_freq)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        torch.save(model.state_dict(), os.path.join(args.save, 'Current_'+str(args.arch)+'.pth'))
        
        if args.save_interval_epoch and (epoch%args.save_interval_epoch==0):
            torch.save(model.state_dict(), os.path.join(args.save, str(args.arch)+'_'+str(epoch)+'_Epoch.pth'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save, 'Best_'+str(args.arch)+'.pth'))



# additional subgradient descent on the sparsity-induced penalty term
# Pruning Code 1
def updateBN(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            #m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
            if m.weight.data.shape[-1]==1:
                m.weight.grad.data.add_(args.s*0.6*torch.sign(m.weight.data))  # L1
            elif m.weight.data.shape[-1]==3:
                L1 = m.weight.data
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
            
def fine_tune(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            mask = (m.weight.data != 0)
            mask = mask.float().to(device)
            m.weight.grad.data.mul_(mask)
        
        elif isinstance(m, nn.Linear):
            mask = (m.weight.data != 0)
            mask = mask.float().to(device)
            m.weight.grad.data.mul_(mask)




def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.sr:     # Pruning Code 3
            updateBN(model) 
        
        if args.fine_tuning:
            fine_tune(model)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        #torch.save(model.state_dict(), os.path.join(args.save, str(args.arch)+'_Baseline.pth'))


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    total = 0
    total_3by3 = 0
    total_1by1 = 0
    state_dict = torch.load(args.resume)
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
    for i in range(0,10):
        print('3by3 Conv L1 Norm', i*10, '% :', y[int(total_3by3 * i*0.1)]/9)
        print()
    print('\n')    
    for i in range(0,10):
        print('1by1 Conv L1 Norm', i*10, '% :', z[int(total_1by1 * i*0.1)])
        print()
    print('\n')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()

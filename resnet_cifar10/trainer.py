import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import resnet_flops
'''
model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
'''

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', default='resnet56')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save', dest='save',
                    help='The directory used to save the trained models',
                    default='weights', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=None)
parser.add_argument('--fine_tuning', dest='fine_tuning', action='store_true')
parser.add_argument('--optimizer', '-o', default='sgd')

best_prec1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    global args, best_prec1, device
    args = parser.parse_args()
    
    if args.arch == 'resnet20':
        net = resnet.resnet20()
    elif args.arch == 'resnet56_flops':
        net = resnet_flops.resnet56()
    elif args.arch == 'resnet110_flops':
        net = resnet_flops.resnet110()
    elif args.arch == 'resnet56':
        net = resnet.resnet56()
    elif args.arch == 'resnet110':
        net = resnet.resnet110()


    # Check the save_dir exists or not
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = torch.nn.DataParallel(net)
    model.cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=0, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            '''
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['weight'])
            first_epoch = checkpoint['epoch']
            print('first_epoch :', first_epoch)
            '''
            if args.evaluate:
                pass
            else:
                best_prec1 = validate(val_loader, model, criterion)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    if args.fine_tuning:
        pass
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)

    '''
    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    '''


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        torch.save(model.state_dict(), os.path.join(args.save, 'Current.pth'))
        #torch.save({'epoch' : first_epoch + epoch, 'weight' : model.state_dict()}, os.path.join(args.save, 'Current.pth'))
        if args.fine_tuning:
            pass
        else:
            lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if prec1 > best_prec1:
            torch.save(model.state_dict(), os.path.join(args.save, 'Best.pth'))
            #torch.save({'epoch': first_epoch+epoch,'weight': model.state_dict()}, os.path.join(args.save, 'Best.pth'))
        best_prec1 = max(prec1, best_prec1)
        if args.save_every and ((first_epoch + epoch)%args.save_every==0):
            torch.save(model.state_dict(), os.path.join(args.save, str(epoch)+'_epoch.pth'))
            #torch.save({'epoch': first_epoch+epoch,'weight': model.state_dict()}, os.path.join(args.save, str(epoch+first_epoch)+'_epoch.pth'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        if args.fine_tuning:
            fine_tune(model)
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


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




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

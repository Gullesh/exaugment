'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from util.misc import CSVLogger
from util.cutout import Cutout
#from util.explain import Explain
from util.gradient import grad
from util.cuex import cuex
from util.ncuex import ncuex
from util.gcam import gcam
from util.gradient import gradbox
from util.re import RandomErasing
from util.cutout import Cutout

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
model_options = ['resnet', 'resnet50','densenet', 'wrn','mobile','dla', 'resnet50']
data_options = ['cifar10', 'cifar100','imagenet']

# Model and Training parameters
parser.add_argument('--model', default='resnet',  choices=model_options)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--data', default='cifar10',  choices=data_options)
# Augmentations and their parameters

parser.add_argument('--re', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--ree', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--explain', action='store_true', default=False,
                    help='apply explain')
parser.add_argument('--gcam', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--gbox', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--sal', action='store_true', default=False,
                    help='if True only keep the salient part')
parser.add_argument('--rescle', action='store_true', default=False,
                    help='scaling random erasing')
parser.add_argument('--cuex', action='store_true', default=False,
                    help='apply cut+explain')
parser.add_argument('--ncuex', action='store_true', default=False,
                    help='apply cut+explain')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--nworkers', type=int, default=2,
                    help='number of workers for trainloader')
parser.add_argument('--pexp', type=float, default=0.25, help='chance of explainablity augmentation happening')
parser.add_argument('--intval', type=float, default=0.8, help='what percentile of lowest intensity pixels to stay')
parser.add_argument('--stepoch', type=int, default=0,
                    help='starting explain augmentation epoch')
parser.add_argument('--a', nargs="+", type=int,help= 'the range of image areas(%) to be removed')
args = parser.parse_args()
print(args)
#torch.multiprocessing.set_start_method('spawn',force=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')


if args.data=='cifar10':
    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # ,value=[0.4914, 0.4822, 0.4465]
    if args.re:
        train_transform.transforms.append(transforms.RandomErasing(p=0.5,rescle=args.rescle))
    if args.cutout:
        train_transform.transforms.append(Cutout(p=args.pexp, rescle=args.rescle))

    transform_test = transforms.Compose([
        transforms.ToTensor(),normalize])
    n_class = 10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers)
elif args.data == 'cifar100':
    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # ,value=[0.4914, 0.4822, 0.4465]
    if args.re:
        train_transform.transforms.append(transforms.RandomErasing(p=0.5,rescle=args.rescle))
    if args.cutout:
        train_transform.transforms.append(Cutout(p=args.pexp, rescle=args.rescle))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ])
    n_class = 100

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers)
# Model
print('==> Building model..')
if args.model == 'resnet':
    net = ResNet18(num_classes=n_class)
    target_layers = [net.layer4[-1]]
elif args.model == 'wrn':
    net = wrn(num_classes=n_class)
    target_layers = [net.block3.layer[-1]]
elif args.model == 'mobile':
    net = MobileNetV2(num_classes=n_class)
    target_layers = [net.layers[16]]

elif args.model == 'dla':
    net = DLA(num_classes=n_class)
    target_layers = [net.layer6]
elif args.model == 'densenet':
    net = DenseNet121(num_classes=n_class)
    target_layers = [net.dense4[15]]
elif args.model == 'resnet50':
    net = ResNet50(num_classes=n_class)
    target_layers = [net.layer4[-1]]
else:
    print('Error: please choose a valid model')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if (args.explain and epoch>=args.stepoch):
            inputs = grad(net,inputs,targets,p=args.pexp, intval = args.intval, sal=args.sal,rescle=args.rescle)
        elif (args.gcam and epoch>=args.stepoch):
            inputs = gcam(net,inputs,targets,target_layers, p=args.pexp, intval = args.intval,sal=args.sal)
        elif (args.cuex and epoch>=args.stepoch):
            inputs = cuex(net,inputs,targets,p=args.pexp, length = args.length,rescle=args.rescle,arng=args.a)
        elif (args.gbox and epoch>=args.stepoch):
            inputs = gradbox(net,inputs,targets,p=args.pexp, length = args.length,rescle=args.rescle, arng=args.a,sal=args.sal)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
  

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

print('best accuracy: ', best_acc)

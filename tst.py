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
from util.gradient import gb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
model_options = ['resnet', 'resnet50','vgg', 'wrn','mobile','dla', 'dpn']
data_options = ['cifar10', 'cifar100','imagenet']

# Model and Training parameters
parser.add_argument('--model', default='resnet',  choices=model_options)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--data', default='cifar10',  choices=data_options)
# Augmentations and their parameters

parser.add_argument('--re', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--ree', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--explain', action='store_true', default=False,
                    help='apply explain')
parser.add_argument('--gcam', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--gbox', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--cutout', action='store_true', default=False,
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
print(args.epoch)
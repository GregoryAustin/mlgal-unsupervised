'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import encoder

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import resnet
import fits_loader
from utils import progress_bar

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import DankNet


######################
# TEST
######################

galaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/galaxies'
nongalaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/nongalaxies'
fitsdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/fits'

transform_train = transforms.Compose([
    fits_loader.RandomCrop(96)
])



galaxhelper = fits_loader.GalaxyHelper(fits_dir=fitsdir, galax_dir=galaxdir, nongalax_dir=nongalaxdir) 
trainset = fits_loader.GalaxyDataset2(galaxhelper, transform_train) 
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)


print("Reading data...")
datasets = []
for x in range(len(trainset)):
    datasets.append(trainset[x])
print("Done. ")


# for batch_idx, (inputs, targets) in enumerate(trainloader):
#     print('Shape: ' + str(inputs.shape) + ' type: ' + str(targets))

######################
# TEST
######################
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
import numpy as np 
import resnet
import fits_loader
from utils import progress_bar

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import DankNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # 0.1 with SGD is best / 1e-2 with adam optimizer was good
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




# DONE: print out 256 * 256 images and check how they looking (galaxies might still not be visible because of the output)
# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# fitsDir = '/home/greg/Desktop/LabelledData/NN project/galaxies/'
# galax = '/home/greg/Desktop/LabelledData/NN project/all_fits.dat'

# galaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/galaxies'
# nongalaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/nongalaxies'
# fitsdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/fits'

# # DONE: normalize 

# transform_train = transforms.Compose([
#     fits_loader.RandomCrop(96)
# ])

# datasets = []

# # trainset1 = fits_loader.GalaxyDataset(fitsDir, galax, transform_train) # TODO: split the train set and test set 
# # datasets.append(trainset1)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



# galaxhelper = fits_loader.GalaxyHelper(fits_dir=fitsdir, galax_dir=galaxdir, nongalax_dir=nongalaxdir) 
# trainset2 = fits_loader.GalaxyDataset2(galaxhelper, transform_train) 

# # this avoids loading multiple FITS files into memory at once
# # and causing the program to MemoryError 
# print("Reading data...")
# dset = []
# for x in range(len(trainset2)):
#     dset.append(trainset2[x])
# print("Done. ")

# datasets.append(dset)

# trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=1, shuffle=False, num_workers=2)




# classes = ('galaxy', 'not-galaxy')

# Model
print('==> Building model..')
# net = VGG('VGG19')

encoder = encoder.Encoder()
encoder.load_state_dict(torch.load('saved_models/encoder.pt'))

# net = resnet.ResNet34(encoder)
# net = net.float()
net = DankNet.DankNet()
net = net.float()

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if device == 'cuda':
    net.cuda()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# criterion = nn.SoftMarginLoss()
# criterion = nn.MarginRankingLoss()
# criterion = nn.NLLLoss()
if device == 'cuda':
    criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr) # weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scaler = StandardScaler() #joblib.load('saved_models/ae_scaler.pkl') # loading saved scaler for normalization !! 

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # data normalize 
        # print(batch_idx)
        # normalizes the data 
        # for x in inputs:

        #     if (np.all(np.isfinite(x[0])) == True):
        #         scaler.partial_fit(x[0]) # 0 because there is only one dimension
        #     else: 
        #         print("Yo " + str(batch_idx) + " is messed up !!!")
        
        # for x in inputs:
        #      if (np.all(np.isfinite(x[0])) == True):
        #         x[0] = torch.from_numpy(scaler.transform(x[0])) 


        inputs = Variable(inputs)
        targets = Variable(targets)
        if device == 'cuda':
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0] # yes
        _, predicted = outputs.max(1) # this is okay I think 
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item() OLD
        correct += predicted.eq(targets).sum().data[0]

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch): # TODO: uncomment when you've done TODO: split dataset into test/train
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        if device == 'cuda':
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data[0]

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
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

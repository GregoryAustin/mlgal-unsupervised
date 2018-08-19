'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import resnet
import fits_loader
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




# TODO: print out 256 * 256 images and check how they looking (galaxies might still not be visible because of the output)
# Data
print('==> Preparing data..')

fitsDir = '/home/greg/Desktop/LabelledData/NN project/galaxies/'
galax = '/home/greg/Desktop/LabelledData/NN project/all_fits.dat'

# TODO: normalize 

transform_train = transforms.Compose([
    fits_loader.RandomCrop(256)
])

trainset = fits_loader.GalaxyDataset(fitsDir, galax, transform_train) # TODO: split the train set and test set 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('galaxy', 'not-galaxy')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = resnet.ResNet18()
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
if device == 'cuda':
    criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = inputs.to(device), targets.to(device)
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

# def test(epoch): # TODO: uncomment when you've done TODO: split dataset into test/train
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     # with torch.no_grad():
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
#         if device == 'cuda':
#             inputs = inputs.cuda()
#             targets = targets.cuda()

#         outputs = net(inputs)
#         loss = criterion(outputs, targets)

#         test_loss += loss.data[0]
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().data[0]

#         progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t7')
#         best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    # test(epoch)

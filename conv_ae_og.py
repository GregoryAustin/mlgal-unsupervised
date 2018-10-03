from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import fits_loader
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import encoder
import cifarencoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from utils import progress_bar

parser = argparse.ArgumentParser(description='VAE MNIST Example')

## COMMENT: original batch size was 128, changed to 32 for small dataset
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img-interval', type=int, default=5000, metavar='N',
                    help='how many batches to wait before saving images')
parser.add_argument('--ds', type=int, default = 1, help='dataset: 1: galaxy, 2: CIFAR10')

# parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                   # help='how big is z')
# parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
#                     help='how big is linear around z')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1} if args.cuda else {}

##################################################################################
#                           DATA GETS LOADED HERE
##################################################################################
# print('==> Preparing data..')


if args.ds == 1:
    fitsDir = '/media/greg/Main Disk 2/UNLABELED DATA/bulge'

    fitsDir = '/home/greg/Desktop/Galaxyfits'



    data_transform = transforms.Compose([
            fits_loader.RandomCrop(96)
        ])


    # DONE: ADD TRANSFORMS

    fitshelper = fits_loader.FitsHelper(root_dir=fitsDir)
    fits_dataset = fits_loader.FitsDataset(root_dir=fitsDir, fitshelper=fitshelper, transform=data_transform)


    fitsTestDir = '/home/greg/Desktop/Galaxyfitstest'
    fitsTesthelper = fits_loader.FitsHelper(root_dir=fitsTestDir)
    fits_testset = fits_loader.FitsDataset(root_dir=fitsTestDir, fitshelper=fitsTesthelper, transform=data_transform)


    train_loader = torch.utils.data.DataLoader(fits_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(fits_testset, batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.ds == 2:

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


##################################################################################
#                          END DATA LOAD
##################################################################################


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        if args.ds == 1:
            encoda = encoder.Encoder(True)
        elif args.ds == 2:
            encoda = cifarencoder.Encoder(True)

        self.encoder = encoda

        # Latent space
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0)
        if args.ds == 1:
            self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0)
        elif args.ds == 2:
            self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        
        self.relu = nn.ReLU()

        self.indices = None

        # how big is out 1  torch.Size([1, 8, 48, 48])
        # how big is out 2  torch.Size([1, 16, 96, 96])
        # how big is out 3  torch.Size([1, 8, 96, 96])
        # how big is out 4  torch.Size([1, 1, 96, 96])

        
    def encode(self, x):
        (out, self.indices) = self.encoder(x)
        return out
    
    def decode(self, z):
        out = z
        
        out = self.relu(self.deconv1(out))
        out = self.unpool(out, self.indices)
        # print("how big is out 1 ",out.size())
        out = self.relu(self.deconv2(out))
        # print("how big is out 2 ",out.size())
        
        # print("how big is out pool ",out.size())
        out = self.relu(self.deconv3(out))
        # print("how big is out 3 ",out.size())
        out = self.deconv4(out)

        
        # print("how big is out 4 ",out.size())
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

model = AE()
model.float()
if args.cuda:
    model.cuda()
#Loss 
MSE = nn.MSELoss(reduce=True)
MSE.cuda()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=5e-4
scaler = StandardScaler()

def train(epoch):
    model.train()
    train_loss = 0

    # lossGraph = []
    counter = 0 
    # change to (data, _) for CIFAR or labeled data
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.float()

        # data = (data-torch.mean(data))/torch.std(data)
        for x in data:
            scaler.partial_fit(x[0]) # 0 because there is only one dimension

        # # # # normalizes the data 
        for x in data:
            x[0] = torch.from_numpy(scaler.transform(x[0])) # new 
        

        data = Variable(data)

        if args.cuda:
            data = data.cuda()

        model.zero_grad()
        output = model(data)
        
        loss = MSE(output, data)
        loss.backward(retain_graph=False)
        optimizer.step()

        train_loss += loss.data[0]

        progress_bar(batch_idx, len(train_loader), 'Loss: %.10f'
            % (loss.data[0] / len(data)))


        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.data[0] / len(data)))

        if batch_idx % args.img_interval == 0:
            tmp = "snapshots/better_auto/" + str(epoch) + '_' + str(counter) + ".png"
            counter += 1
            dt = []
            for x in range(len(data.data.cpu())):
                dt.append(data.data.cpu()[x][0])
                # dt.append(torch.ones(data.data.cpu()[x][0].shape[1],4))
            xy = []
            xy.append(torch.cat((dt), 1))
            xy.append(torch.ones(4, xy[0].shape[1]))

            dt = []
            for x in range(len(output.data.cpu())):
                dt.append(output.data.cpu()[x][0])
                # dt.append(torch.ones(output.data.cpu()[x][0].shape[1],4))

            xy.append(torch.cat((dt), 1))

            plt.imsave(tmp, torch.cat((xy), 0), cmap='gray')

        # lossGraph.append(loss.data[0] / len(data))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))

    

    # plt.plot(lossGraph)
    # plt.ylabel('loss')
    # plt.xlabel('iteration count')
    # plt.show()

    # print(data.data.cpu()[0][0].shape)

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
    # for i, data in enumerate(test_loader):
        data = data.float()
        # for x in data:
        #     scaler.partial_fit(x[0]) # 0 because there is only one dimension

        # # # # normalizes the data 
        # for x in data:
        #     x[0] = torch.from_numpy(scaler.transform(x[0])) # new 

        data = (data-torch.mean(data))/torch.std(data)


        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        loss = MSE(output, data)

        test_loss += loss.data[0]
        # if i % 100 == 0:
        #     n = min(data.size(0), 8)
        #     comparison = torch.cat([data[:n],
        #                            output[:n]])
        #     save_image(comparison.data.cpu(),
        #                'snapshots/conv_vae/reconstruction_' + str(epoch) +
        #                '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# tester(1)
# for epoch in range(1, args.epochs + 1):
#     train(epoch)

#     # if epoch == args.epochs:


#     # TODO: get test working again 
#     test(epoch)
    # if epoch == args.epochs:
    #     sample = Variable(torch.randn(64, args.hidden_size))
    #     if args.cuda:
    #         sample = sample.cuda()
    #     sample = model.decode(sample).cpu()
    #     save_image(sample.data.view(64, 3, 32, 32),
    #                'snapshots/conv_vae/sample_' + str(epoch) + '.png')
    # TODO: this saves the hidden layer, maybe use this at some point 


def trainAndTest(epochs=5, encNum=1):
    for run in range(0, encNum):
        global model, optimizer 
        model = AE()
        # net = DankNet.TrashNet(encoda)
        model = model.float()
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            train(epoch) 
            test(epoch) 
        if args.ds == 1:
            torch.save(model.encoder.state_dict(), 'saved_models/encoder' + str(run) + '.pt')
        if args.ds == 2:
            torch.save(model.encoder.state_dict(), 'cifar_models/encoder' + str(run) + '.pt')

if args.ds == 1:
    trainAndTest(1, 10)
elif args.ds == 2:
    trainAndTest(10,10)
 # saves the autoencoder
# joblib.dump(scaler , 'saved_models/ae_scaler.pkl') # saves the normalized scaler



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
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='VAE MNIST Example')

## COMMENT: original batch size was 128, changed to 32 for small dataset
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving images')
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


kwargs = {'num_workers': 2} if args.cuda else {}

##################################################################################
#                           DATA GETS LOADED HERE
##################################################################################
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



# fitsDir = '/media/greg/Main Disk 2/UNLABELED DATA/bulge'

# fitsDir = '/home/greg/Desktop/Galaxyfits'



# data_transform = transforms.Compose([
#         fits_loader.RandomCrop(256)
#     ])


# # DONE: ADD TRANSFORMS

# fitshelper = fits_loader.FitsHelper(root_dir=fitsDir)
# fits_dataset = fits_loader.FitsDataset(root_dir=fitsDir, fitshelper=fitshelper, transform=data_transform)


# # this avoids loading multiple FITS files into memory at once
# # and causing the program to MemoryError 

# print("Reading data...")
# datasets = []
# for x in range(len(fitshelper.getFits())):
#     idxs = fitshelper.getFitsFileSlice(x)
#     dtaset = []
#     for z in range(len(range(idxs[0], idxs[1]))): # TODO: fix this 
#         dtaset.append(fits_dataset[z])

#     datasets.append(dtaset)
# print("Done. ")

# train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, **kwargs)

#test_loader = torch.utils.data.DataLoader(fits_dataset,
#    batch_size=args.batch_size, shuffle=False, **kwargs)

##################################################################################
#                          END DATA LOAD
##################################################################################


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = encoder.Encoder(True)
        
        # Latent space
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        
        self.relu = nn.ReLU()

        self.indices = None

        self.dense1_bn = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, 10)

        self.soft = nn.Softmax(dim=1)
        
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

    def classify(self, x):
        # (out, _) = self.encoda(x)
        # print(out.size())

        out = x
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.dense1_bn(out) # TODO: try without this
        out = self.fc(out)
        out = self.soft(out)

        return out

    def forward(self, x):
        z = self.encode(x)
        ae_out = self.decode(z)
        classy_out = self.classify(z)

        return (ae_out, classy_out)

model = AE()
model.float()
if args.cuda:
    model.cuda()
#Loss 
MSE = nn.MSELoss(reduce=True)
MSE.cuda()

CSL = nn.CrossEntropyLoss()
CSL.cuda()


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=5e-4
scaler = StandardScaler()

def train(epoch):
    model.train()
    aeLmao_loss = 0
    classyBoi_loss = 0

    # lossGraph = []
    total = 0 
    correct = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.float()


        # for x in data:
        #     scaler.partial_fit(x[0]) # 0 because there is only one dimension

        # # # normalizes the data 
        # for x in data:
        #     x[0] = torch.from_numpy(scaler.transform(x[0])) # new 
        

        data = Variable(data)
        targets = Variable(targets)

        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        model.zero_grad()
        ae_output, classifier_output = model(data)
        
        ae_loss = MSE(ae_output, data)

        ae_loss.backward(retain_graph=True)

        classy_loss = CSL(classifier_output, targets)

        classy_loss.backward(retain_graph=True)

        optimizer.step()

        aeLmao_loss += ae_loss.data[0]

        classyBoi_loss += classy_loss.data[0]

        _, predicted = classifier_output.max(1) # this is okay I think 
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data[0]
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\tCLoss: {:.5f}\tAcc: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ae_loss.data[0] / len(data), classy_loss.data[0] / len(data), 100.*correct/total))

        if batch_idx % 100 == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                   ae_output[:n]])
            save_image(comparison.data.cpu(),
                       'snapshots/conv_vae/reconstruction_' + str(epoch) + str(batch_idx) +
                       '.png', nrow=n)


        
        
        # correct += predicted.eq(targets).sum().item() OLD
        

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # if batch_idx % args.img_interval == 0:
        #     tmp = "snapshots/better_auto/" + str(epoch) + '_' + str(counter) + ".png"
        #     counter += 1
        #     dt = []
        #     for x in range(len(data.data.cpu())):
        #         dt.append(data.data.cpu()[x][0])
        #         dt.append(torch.ones(data.data.cpu()[x][0].shape[1],4))
        #     xy = []
        #     xy.append(torch.cat((dt), 1))
        #     xy.append(torch.ones(4, xy[0].shape[1]))

        #     dt = []
        #     for x in range(len(output.data.cpu())):
        #         dt.append(output.data.cpu()[x][0])
        #         dt.append(torch.ones(output.data.cpu()[x][0].shape[1],4))

        #     xy.append(torch.cat((dt), 1))

        #     plt.imsave(tmp, torch.cat((xy), 0), cmap='gray')

        # lossGraph.append(loss.data[0] / len(data))

    print('====> Epoch: {} Average loss: {:.4f}\tCloss {:.10f}'.format(
          epoch, aeLmao_loss / len(train_loader.dataset), classyBoi_loss / len(train_loader.dataset)))

    

    # plt.plot(lossGraph)
    # plt.ylabel('loss')
    # plt.xlabel('iteration count')
    # plt.show()

    # print(data.data.cpu()[0][0].shape)

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     # for i, (data, _) in enumerate(test_loader):
#     for i, (data, _) in enumerate(test_loader):
#         data = data.float()
#         if args.cuda:
#             data = data.cuda()
#         data = Variable(data, volatile=True)
#         output = model(data)
#         loss = MSE(output, data)

#         test_loss += loss.data[0]
#         if i % 100 == 0:
#             n = min(data.size(0), 8)
#             comparison = torch.cat([data[:n],
#                                    output[:n]])
#             save_image(comparison.data.cpu(),
#                        'snapshots/conv_vae/reconstruction_' + str(epoch) +
#                        '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


# tester(1)
for epoch in range(1, args.epochs + 1):
    train(epoch)

    # if epoch == args.epochs:


    # TODO: get test working again 
    # test(epoch)
    # if epoch == args.epochs:
    #     sample = Variable(torch.randn(64, args.hidden_size))
    #     if args.cuda:
    #         sample = sample.cuda()
    #     sample = model.decode(sample).cpu()
    #     save_image(sample.data.view(64, 3, 32, 32),
    #                'snapshots/conv_vae/sample_' + str(epoch) + '.png')
    # TODO: this saves the hidden layer, maybe use this at some point 

torch.save(model.encoder.state_dict(), 'saved_models/encoder.pt') # saves the autoencoder
# joblib.dump(scaler , 'saved_models/ae_scaler.pkl') # saves the normalized scaler



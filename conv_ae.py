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

fitsDir = '/media/greg/Main Disk 2/UNLABELED DATA/bulge'

fitsDir = '/home/greg/Desktop/Galaxyfits'



data_transform = transforms.Compose([
        fits_loader.RandomCrop(512)
    ])


# DONE: ADD TRANSFORMS
# TODO: RANDOM GAUSSIAN NOISE 
# TODO: RANDOM HORIZONTAL FLIP 
# TODO: RANDOM VERTICAL FLIP 

fitshelper = fits_loader.FitsHelper(root_dir=fitsDir)
fits_dataset = fits_loader.FitsDataset(root_dir=fitsDir, fitshelper=fitshelper, transform=data_transform)


# this avoids loading multiple FITS files into memory at once
# and causing the program to MemoryError 

print("Reading data...")
datasets = []
for x in range(len(fitshelper.getFits())):
    idxs = fitshelper.getFitsFileSlice(x)
    dtaset = []
    for z in range(len(range(idxs[0], idxs[1]))):
        dtaset.append(fits_dataset[z])

    datasets.append(dtaset)
print("Done. ")

train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, **kwargs)

#test_loader = torch.utils.data.DataLoader(fits_dataset,
#    batch_size=args.batch_size, shuffle=False, **kwargs)

##################################################################################
#                          END DATA LOAD
##################################################################################

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder     input size: 1,512,512 (262k)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1) #8,256,256
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)#16,128,128
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)#8,64,64 
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)#4,32,32 

        
        # Latent space
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        out = self.relu(self.conv1(x))
        # print("how big is this 1 ",out.size()) # 8, 64, 64
        out = self.relu(self.conv2(out))
        # print("how big is this 2 ",out.size()) # 16, 128, 128 
        out = self.relu(self.conv3(out))
        # print("how big is this 2 ",out.size())
        out = self.conv4(out)
        # print("how big is this 3 ",out.size())
        return out

    
    def decode(self, z):
        out = self.relu(self.deconv1(z))
        # print("how big is this 1 ",out.size())
        out = self.relu(self.deconv2(out))
        # print("how big is this 2 ",out.size())
        out = self.relu(self.deconv3(out))
        # print("how big is this 3 ",out.size())
        out = self.deconv4(out)
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    train_loss = 0

    # lossGraph = []
    # # # for batch_idx, (data, _) in enumerate(train_loader):
    # maxs = []
    # mins = []
    # means = []

    min_x = -142.305237
    max_x = 41916.58203125

    counter = 0 

    for batch_idx, data in enumerate(train_loader):
        data = data.float()

        # normalize stuff 
        # data = (data-torch.mean(data))/torch.std(data) OLD
        # data = torch.sqrt(data + 2453) OLD 

        # maxs.append(torch.max(data)) FOR CHECKING MIN MAX MEAN
        # means.append(torch.mean(data))
        # mins.append(torch.min(data))
        # normalize stuff

        # normalizes the data 
        data = (data - min_x) / (max_x - min_x)
        

        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        model.zero_grad()
        output = model(data)
        
        loss = MSE(output, data)
        loss.backward(retain_graph=False)
        optimizer.step()

        train_loss += loss.data[0]

        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

        if batch_idx % args.img_interval == 0:
            tmp = "snapshots/better_auto/" + str(epoch) + '_' + str(counter) + ".png"
            counter += 1
            dt = []
            for x in range(len(data.data.cpu())):
                dt.append(data.data.cpu()[x][0])
                dt.append(torch.ones(data.data.cpu()[x][0].shape[1],4))
            xy = []
            xy.append(torch.cat((dt), 1))
            xy.append(torch.ones(4, xy[0].shape[1]))

            dt = []
            for x in range(len(output.data.cpu())):
                dt.append(output.data.cpu()[x][0])
                dt.append(torch.ones(output.data.cpu()[x][0].shape[1],4))

            xy.append(torch.cat((dt), 1))

            plt.imsave(tmp, torch.cat((xy), 0), cmap='gray')

        # lossGraph.append(loss.data[0] / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    # print("MAX: " + str(max(maxs)))
    # print("MIN: ", min(mins))
    # print("MEANS: ", sum(means)/len(means))

    # plt.plot(lossGraph)
    # plt.ylabel('loss')
    # plt.xlabel('iteration count')
    # plt.show()

    # print(data.data.cpu()[0][0].shape)

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     # for i, (data, _) in enumerate(test_loader):
#     for i, data in enumerate(test_loader):
#         data = data.float()
#         if args.cuda:
#             data = data.cuda()
#         data = Variable(data, volatile=True)
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#         if epoch == args.epochs and i == 0:
#             n = min(data.size(0), 8)
#             comparison = torch.cat([data[:n],
#                                    recon_batch[:n]])
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




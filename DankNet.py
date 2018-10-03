import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import encoder

class DankNet(nn.Module):
    def __init__(self, galaxy=True):
        super(DankNet, self).__init__()
        # input is 1 * 96 * 96
        self.galaxy = galaxy
        self.relu = nn.ReLU()
        self.noise = GaussianNoise(0.01) 
        if galaxy: 
            self.down1 = same(1, 32)
        else:
            self.down1 = same(3, 32) # CIFAR 10
        
        self.pool = nn.MaxPool2d(2, stride=2)

        if not galaxy:
            self.conv2_bn = nn.BatchNorm2d(32) # CIFAR 10

        self.down2 = same(32, 64)
        
        self.down3 = same(64, 128)
        
        self.down4 = same(128, 256)
        self.down5 = same(256, 512)

        self.fc1 = one(512,128)

        # self.dropout = DropOut(1)
        # self.fc2 = one(128, 32)

        self.dense1_bn = nn.BatchNorm1d(128)

        if galaxy:
            self.fc3 = nn.Linear(128, 2, bias=False) # GALAXY
        else:
            self.fc3 = nn.Linear(128, 10, bias=False) # CIFAR 10

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # out = self.gauss(x)
        # (out, _) = self.encoder(x)

        out = self.noise(x)

        out = self.down1(out)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        # out = self.dropout(out)
    
       
        out = self.down2(out)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        # out = self.dropout(out)
        # print(out.size())

        out = self.down3(out)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        # out = self.dropout(out)
        # print(out.size())

        out = self.down4(out)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        # out = self.dropout(out)
        # print(out.size())

        out = self.down5(out)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        # out = self.dropout(out)
        # print(out.size())

        if self.galaxy:
            out = torch.mean(out, dim=2, keepdim=True) # this wasnt in cifar 10 
            # print(out.size())

            out = torch.mean(out, dim=3, keepdim=True) # this wasnt in cifar 10 
            # print(out.size())

        out = self.relu(self.fc1(out))
        # print(out.size())
        # out = self.fc2(out)
        # print(out.size())
        out = out.view(out.size()[0], -1)
        # print(out.size())
        if not self.galaxy:
            out = self.dense1_bn(out)
        
        out = self.fc3(out)
        # print(out.size())
        # print(out.size())

        out = self.soft(out)

        return out


# SOME KEY DIFFERENCES: 
# DankEncodeNet with an Encoder for CIFAR 10 is missing layer 4/5 because of dimensionality
# DankEncodeNet with an Encoder for Galaxies is missing pooling layer 5 because of dimensionality

class DankEncodeNet(nn.Module):
    def __init__(self, encoda, galaxy):
        super(DankEncodeNet, self).__init__()

        self.galaxy = galaxy
        # input is 1 * 96 * 96
        # input is 4 * 8 * 8

        self.encoda = encoda
        self.relu = nn.ReLU()
        self.down1 = same(4, 32)
        
        self.pool = nn.MaxPool2d(2, stride=2)

        # self.conv2_bn = nn.BatchNorm2d(32)
        self.down2 = same(32, 64)
        
        self.down3 = same(64, 128)
        
        self.down4 = same(128, 256)
        self.down5 = same(256, 512, False)

        self.fc1 = one(512,128)
        # self.fc2 = one(128, 32)

        self.dense1_bn = nn.BatchNorm1d(128)

        if galaxy:
            self.fc3 = nn.Linear(128, 2, bias=False)
        else:
            self.fc3 = nn.Linear(128, 10, bias=False)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # out = self.gauss(x)
        (out, _) = self.encoda(x)
        # print(out.size())

        out = self.down1(out)
        out = self.pool(out)
        # print(out.size())
       
        out = self.down2(out)
        out = self.pool(out)
        # print(out.size())

        out = self.down3(out)
        out = self.pool(out)
        # print(out.size())
        if self.galaxy:
            out = self.down4(out)
            out = self.pool(out)
            # print(out.size())

            out = self.down5(out)
            # out = self.pool(out) NO POOLING THIS LAYER
            # print(out.size())

            out = self.relu(self.fc1(out))
        # print(out.size())
        # out = self.fc2(out)
        # print(out.size())
        out = out.view(out.size()[0], -1)
        # print(out.shape)

        if not self.galaxy:
            out = self.dense1_bn(out)
        
        out = self.fc3(out)
        # print(out.size())
        # print(out.size())

        out = self.soft(out)

        return out

class TrashEncodeNet(nn.Module):
    def __init__(self, encoda, galaxy):
        super(TrashEncodeNet, self).__init__()
        self.encoda = encoda # 4, 8, 8
        self.galaxy = galaxy

        self.dense1_bn = nn.BatchNorm1d(256) # USED 256 for CIFAR 10 # USED 2304 for galaxies
        
        if galaxy:
            self.fc = nn.Linear(2304, 2)
        else: 
            self.fc = nn.Linear(256, 10)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        (out, _) = self.encoda(x)
        # print(out.size())

        
        out = out.view(out.size(0), -1)
        # print(out.size())
        if not self.galaxy:
            out = self.dense1_bn(out) # TODO: try without this
        out = self.fc(out)
        out = self.soft(out)

        return out

class TrashNet(nn.Module):
    def __init__(self, galaxy):
        super(TrashNet, self).__init__()

        self.galaxy = galaxy
        if galaxy:
            self.dense1_bn = nn.BatchNorm1d(9216) # CIFAR USED 3072
            self.fc = nn.Linear(9216, 2)
        else:
            self.dense1_bn = nn.BatchNorm1d(3072) # CIFAR USED 3072
            self.fc = nn.Linear(3072, 10)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # (out, _) = self.encoda(x)
        # print(out.size())

        out = x
        out = out.view(out.size(0), -1)
        # print(out.size())
        if not self.galaxy:
            out = self.dense1_bn(out) # (CIFAR 10 used this)
        
        out = self.fc(out)
        out = self.soft(out)

        return out        

class DropOut(nn.Module):
    def __init__(self,drp):
        super().__init__()
        self.drp = nn.Dropout(0.1)  

    def forward(self, din):
        if self.training:
            return self.drp(din)
        # print("Its not training nice")
        return din

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


def test():
    # encoda/ = encoder.Encoder(False)
    # encoda.load_state_dict(torch.load('saved_models/encoder.pt'))

    # net = resnet.ResNet34(encoder)
    # net = net.float()

    net = TrashNet()
    # net = DankNet.DankNet()
    net = net.float()

    y = net(Variable(torch.randn(1,1,96,96)))
    # print(y.size())


def testDank():
    net = DankNet().cuda()
    net = net.float()
    y = net(Variable(torch.randn(8,3,32,32)).cuda())


class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU() #inplace=True
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class same(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(same, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        self.norm = norm
        # self.sqx = SqEx(out_channels)
        #self.noise = GaussianNoise(0.01)
        self.residual = False
        if in_channels == out_channels:
            self.residual = True

    def forward(self,x):
        a = self.conv(x)
        #a = self.noise(a)
        a = self.relu(a)
        if (self.norm):
            a = self.lrn(a)
        # a = self.sqx(a)
        if self.residual:
            a = (a + x)*0.5
        return a

class down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(down, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        self.sqx = SqEx(out_channels)

    def forward(self,x):
        a = self.conv(x)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        return a

class one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(one, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        # self.sqx = SqEx(out_channels)

    def forward(self,x):
        a = self.conv(x)
        a = self.relu(a)
        # a = self.lrn(a)
        # a = self.sqx(a)
        return a

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.conv = nn.Conv2d(
        #    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.lrn = nn.BatchNorm2d(out_channels)
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sqx = SqEx(out_channels)

    def forward(self,x):
        #a = self.upsample(x)
        a = self.conv(x)
        a = self.relu(a)
        a = self.lrn(a)
        a = self.sqx(a)
        return a

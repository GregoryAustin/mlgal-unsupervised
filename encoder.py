import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder (nn.Module):
    def __init__(self, withGauss):
        super(Encoder,self).__init__()
        self.withG = withGauss
        self.gauss = GaussianNoise(0.01)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1) 

#         how big is this 1  torch.Size([1, 8, 96, 96])
#         how big is this 2  torch.Size([1, 16, 96, 96])
#         how big is this 3  torch.Size([1, 8, 48, 48])
#         how big is this pool  torch.Size([1, 8, 24, 24])
#         how big is this 4  torch.Size([1, 4, 24, 24])



        # 4 * 8 * 8 

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.withG:
            out = self.gauss(x)
        else:
            # print('not using guass')
            out = x

        out = self.relu(self.conv1(out))
        # print("how big is this 1 ",out.size()) 
        out = self.relu(self.conv2(out))
        # print("how big is this 2 ",out.size())

        
        # print("how big is this pool ",out.size())

        out = self.relu(self.conv3(out))
        # print("how big is this 3 ",out.size())
        (out, indices) = self.pool(out)
        
        # print("how big is this pool ",out.size())
        out = self.conv4(out)
        # print(out.size())
        
        # print("how big is this 4 ",out.size())
        return (out, indices)

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

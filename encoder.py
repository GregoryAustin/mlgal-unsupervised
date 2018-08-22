import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder (nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=2, padding=1) #8,256,256 # 8 * 128 * 128 # 6 * 128 * 128
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)#16,128,128 # 16 * 64 * 64 # 12 * 64 * 64
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3, stride=2, padding=1)#8,64,64 # 8 * 32 * 32 # 6 * 32 * 32
        self.conv4 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=0)#4,32,32 # 4 * 32 * 32 # 3 * 32 * 32

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        # print("how big is this 1 ",out.size()) # 8, 64, 64
        out = self.relu(self.conv2(out))
        # print("how big is this 2 ",out.size()) # 16, 128, 128 
        out = self.relu(self.conv3(out))
        # print("how big is this 2 ",out.size())
        out = self.conv4(out)
        # print("how big is this 3 ",out.size())
        return out


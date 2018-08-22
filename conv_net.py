import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder # getting the encoder from the trained autoencoder

class ConvNet(nn.Module):
    def __init__(self, encoder, numclasses=2):
        super(ConvNet, self).__init__()
        # THIS SUCKA IS GETTING A 1 * 256 * 256 BATCHED 16
        self.encoder = encoder
        self.conv1 = nn.Conv2d(3,6,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear()




    def forward(self, x):
        out = self.encoder(x)

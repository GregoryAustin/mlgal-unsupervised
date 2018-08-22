import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder # getting the encoder from the trained autoencoder

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()



    def forward(self, x):
        

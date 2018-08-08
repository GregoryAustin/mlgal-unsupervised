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
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy
import os


class FitsDataset(Dataset):
    """Face Landmarks dataset."""
    # make this more efficient 
    # be able to know all files but also 
    # be able to index without reading every file in


    def __init__(self, root_dir, dimensions=554, transform=None):
        
        fitsFiles = os.listdir(root_dir)

        
        # TODO: make it read ALL fits files, not just this one

        self.fits_files = fitsFiles
        print("Reading fits files and storing dimensions for efficiency... ")
        self.dimensions = dimensions

        dim_arr = []
        for x in fitsFiles:
            data = fits.getdata(root_dir + "/" + x)
            x = data.shape[0] - (data.shape[0] % dimensions)
            y = data.shape[1] - (data.shape[1] % dimensions)
            dim_arr.append((x,y))

        self.fits_dimensions = dim_arr
        print("Done.")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # calculating length is a bit more complicated 
        return 200


# (15540, 12765)
# (15540, 12765)
# (15540, 12765)
# (15540, 12765)
# (15540, 12765)


    def __getitem__(self, idx):
        # given an index i
        # given x*x for crop image dimensions 
        # given n files of known dimensions
        # i * x/2 = pixel_value (divided by 2 because of horizontal interleaving) 
        # pixel_value/width = rows remainder pixels
        # fits_rows / rows = file_number remainder rows 
        # there is vertical interleaving and horizontal interleaving 
        #                       /2 on both  
        
        pixels = index * self.dimensions/2
        
        row = int(pixels/self.fits_dimensions[0][0]) # row number
        totalRows = self.fits_dimensions[0][1] / self.dimensions/2
        print(totalRows)

        totalPixels = self.fits_dimensions[0][1] # start initially with 
        # while (pixels > totalPixels):


        pixels = pixels % width # x value for pixels

        #TODO: test for end and beginning bounds on fits files 


        tmpFile = root_dir + '/' + fitsFiles[0]
        tmpFile = fits.getdata(tmpFile, ext=0)

        # THIS CODE CROPS FITS FILES AND RETURNS A SINGLE CHANNEL TENSOR 
        #         crop_image = tmpFile[x:x+whatever, y:y+whatever]
                
        #         crop_image = crop_image[..., numpy.newaxis]
        #         crop_image = crop_image.transpose(2, 0, 1)

        #         tensor = torch.from_numpy(crop_image)

        sample = self.fits_file[idx]

        

        if self.transform:
            sample = self.transform(sample)


        #TODO: all transform things that are cool 
        #TODO: normalize dataset

        return sample


    # normalize tensors!!  
    # this is a normalize function 
    def normTensor(x):
        return (x-torch.mean(x))/torch.std(x)
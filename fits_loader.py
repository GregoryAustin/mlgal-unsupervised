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
import pandas as pd
import numpy
import os

default_dimens = 530


class FitsHelper():
    def __init__(self, root_dir, dimensions=default_dimens, galaxies=None):
        fitsFiles = os.listdir(root_dir)
        print("Reading fits files and storing dimensions for efficiency and logic... ")
        self.fitsFiles = fitsFiles 

        dim_arr = []
        for x in fitsFiles: 
            data = fits.getdata(root_dir + "/" + x)
            # print (data.shape) # TEST
            x = data.shape[0] - (data.shape[0] % dimensions)
            y = data.shape[1] - (data.shape[1] % dimensions)
            dim_arr.append((x,y))

        self.galaxies = galaxies
        if (self.galaxies):
            self.galaxies = pd.read_csv(self.galaxies, delim_whitespace=True)

        self.fits_dims = dim_arr 
        self.dimensions = dimensions
        print("Done.")

    def getFits(self):
        return self.fitsFiles 

    def getGalaxies(self):
        # print(self.galaxies.iloc[:, 0])
        counter = 0 
        for x in range(len(self.galaxies)):
            for y in range(len(self.fitsFiles)):
                if self.galaxies.iloc[x, 0] in self.fitsFiles[y]:
                    counter += 1 
                    break

        print(counter) 
            # print(galaxy_name)



    def getFitsDimensions(self):
        return self.fits_dims

    # this returns the indices in the dataset belonging to one fits file 
    def getFitsFileSlice(self, idx):
        total = 0
        prevTotal = 0
        for x in range(idx+1):
            prevTotal = total 
            totalRows = self.fits_dims[x][1] / (self.dimensions/2)
            total += (totalRows-1) * (self.fits_dims[x][0]/(self.dimensions/2))
        
        return (int(prevTotal), int(total))

#############################################
# TESTING 

fitsDir = '/home/greg/Desktop/LabelledData/NN project/galaxies/'
galax = '/home/greg/Desktop/LabelledData/NN project/2M_col_corr_comb_param_gal.dat'

fitshelper = FitsHelper(fitsDir, galaxies = galax)
fitshelper.getGalaxies()

# TESTING 
#############################################


class FitsDataset(Dataset):
    """Face Landmarks dataset."""
    # make this more efficient 
    # be able to know all files but also 
    # be able to index without reading every file in

    # DONE: would prefer dimensions to be 530 and then randomcrop! 
    # TODO: add noise to images 
    # DONE: normalize dataset

    # TODO: CNN 
    def __init__(self, root_dir, fitshelper, dimensions=default_dimens, transform=None):
        
        self.fits_files = fitshelper.getFits() # NEW

        self.dimensions = dimensions

        self.curr_fits = 0
        self.data = fits.getdata(root_dir + "/" + self.fits_files[0])

        self.fits_dimensions = fitshelper.getFitsDimensions()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        total = 0

        for x in range(len(self.fits_files)):
            totalRows = self.fits_dimensions[x][1] / (self.dimensions/2)
            total += (totalRows-1) * (self.fits_dimensions[x][0]/(self.dimensions/2))
        
        return int(total)
        # return 2000


    def __getitem__(self, idx):
        # given an index i
        # given x*x for crop image dimensions 
        # given n files of known dimensions
        # i * x/2 = pixel_value (divided by 2 because of horizontal interleaving) 
        # pixel_value/width = rows remainder pixels
        # fits_rows / rows = file_number remainder rows 
        # there is vertical interleaving and horizontal interleaving 
        #                       /2 on both  
        
        pixels = idx * self.dimensions/2
        

        # Get correct fits file 
        fitsFile = 0
        # initializing these for the while loop 
       
        row = int(pixels/self.fits_dimensions[fitsFile][0]) # row number
        totalRows = self.fits_dimensions[fitsFile][1] / (self.dimensions/2)
        while (row > totalRows-2):
            pixels -= (totalRows-1)*self.fits_dimensions[fitsFile][0]
            fitsFile += 1

            row = int(pixels/self.fits_dimensions[fitsFile][0]) # row number
            totalRows = self.fits_dimensions[fitsFile][1] / (self.dimensions/2)

        # print("This is for fits: " + str(fitsFile))

        x = int(pixels % self.fits_dimensions[fitsFile][0]) # x value for pixels
        y = int(row*self.dimensions/2) # y value for pixels

        if (self.curr_fits == fitsFile):
            tmpFile = self.data
        else:
            tmpFile = self.root_dir + '/' + self.fits_files[fitsFile]
            tmpFile = fits.getdata(tmpFile, ext=0) 
            self.curr_fits = fitsFile
            self.data = tmpFile 
        

        # THIS CODE CROPS FITS FILES AND RETURNS A SINGLE CHANNEL TENSOR 
        crop_image = tmpFile[x:x+self.dimensions, y:y+self.dimensions]
        
        if self.transform:
            crop_image = self.transform(crop_image)

        # Converting to one channel tensor 
        crop_image = crop_image[..., numpy.newaxis]
        crop_image = crop_image.transpose(2, 0, 1)

        sample = torch.from_numpy(crop_image)
        
        return sample


# 0.1 = 10% 
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(0, self.stddev).cuda()
            return x + noise 
        else:
            return x




class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = numpy.random.randint(0, h - new_h)
        left = numpy.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        
        return image
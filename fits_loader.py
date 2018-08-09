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

    # TODO: would prefer dimensions to be 530 and then randomcrop! 
    # TODO: get fits files that are h,j,k !! 
    # TODO: figure out how to shuffle without having to constantly open and 
    # close fits files
    # TODO: learn how to add logistic regression to autoencoder 
    #  or possibly something else like a CNN? 
    def __init__(self, root_dir, dimensions=512, transform=None):
        
        fitsFiles = os.listdir(root_dir)

        
        # DONE: make it read ALL fits files, not just this one

        self.fits_files = fitsFiles
        print("Reading fits files and storing dimensions for efficiency and logic... ")
        self.dimensions = dimensions

        dim_arr = []
        for x in fitsFiles:
            data = fits.getdata(root_dir + "/" + x)
            x = data.shape[0] - (data.shape[0] % dimensions)
            y = data.shape[1] - (data.shape[1] % dimensions)
            dim_arr.append((x,y))


        self.curr_fits = 0
        self.data = fits.getdata(root_dir + "/" + fitsFiles[0])

        self.fits_dimensions = dim_arr
        print("Done.")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        total = 0

        for x in range(len(self.fits_files)):
            totalRows = self.fits_dimensions[x][1] / (self.dimensions/2)
            total += (totalRows-1) * (self.fits_dimensions[x][0]/(self.dimensions/2))
            # print(total)
        
        #print("TOTAL" + str(total))
        return int(total)
        # return 2000


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
            tmpFile = fits.getdata(tmpFile, ext=0) # calling this fucks shuffle up
            self.curr_fits = fitsFile
            self.data = tmpFile 
        

        # THIS CODE CROPS FITS FILES AND RETURNS A SINGLE CHANNEL TENSOR 
        crop_image = tmpFile[x:x+self.dimensions, y:y+self.dimensions]
        
        crop_image = crop_image[..., numpy.newaxis]
        crop_image = crop_image.transpose(2, 0, 1)

        sample = torch.from_numpy(crop_image)

        if self.transform:
            sample = self.transform(sample)


        #TODO: all transform things that are cool 
        #TODO: normalize dataset
        # print((x,y))

        return sample


    # TODO: normalize tensors!!  
    # this is a normalize function 
    def normTensor(x):
        return (x-torch.mean(x))/torch.std(x)

# fitsDir = '/home/greg/Desktop/Galaxyfits'

# fits_dataset = FitsDataset(root_dir=fitsDir)

# print(len(fits_dataset))

# for i in range(len(fits_dataset)):
#     sample = fits_dataset[i]


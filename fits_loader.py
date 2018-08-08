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


# ##############################################################################
# # Display the image data:




class FitsDataset(Dataset):
    """Face Landmarks dataset."""
    # make this more efficient 
    # be able to know all files but also 
    # be able to index without reading every file in


    def __init__(self, root_dir, dimensions=544, transform=None):
        
        fitsFiles = os.listdir(root_dir)

        
        # TODO: make it read ALL fits files, not just this one
        

        # img_size = 544
        # tensors = []
        # prev_x = 0
        # ## CAUTION: THIS CROPS OFF SOME OF THE FITS DATA FILES 
        # for x in range(1, int(tmpFile.shape[0]/img_size)): 
        #     for y in range(1, int(tmpFile.shape[1]/img_size)-1):
        #         crop_image = tmpFile[prev_x:x*img_size, y*img_size:(y*img_size + img_size)]
                
        #         crop_image = crop_image[..., numpy.newaxis]
        #         crop_image = crop_image.transpose(2, 0, 1)
        #         # TODO: Give value to new dimension?? 
        #         # print(crop_image.dtype)

        #         tensors.append(torch.from_numpy(crop_image))
        #     prev_x = x*img_size

        self.fits_files = fitsFiles
        self.dimensions = dimensions
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fits_file)

    def fancyIndex(self, index):
    	# given an index i
    	# given x*x for crop image dimensions 
    	# given n files of known dimensions
    	# i * x/2 = pixel_value (divided by 2 because of horizontal interleaving) 
    	# pixel_value/width = rows remainder pixels
    	# fits_rows / rows = file_number remainder rows 
    	# there is vertical interleaving and horizontal interleaving 
    	# 						/2 on both  
    	pixels = index * self.dimensions/2
    	width = 
    	row = pixels/width



    	tmpFile = root_dir + '/' + fitsFiles[0]
        tmpFile = fits.getdata(tmpFile, ext=0)

    def __getitem__(self, idx):
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
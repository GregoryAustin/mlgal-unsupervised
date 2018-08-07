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

    def __init__(self, root_dir, transform=None):
        
        fitsFiles = os.listdir(root_dir)
        
        # TODO: make it read ALL fits files, not just this one
        tmpFile = root_dir + '/' + fitsFiles[0]
        tmpFile = fits.getdata(tmpFile, ext=0)

        img_size = 512
        tensors = []
        prev_x = 0
        ## CAUTION: THIS CROPS OFF SOME OF THE FITS DATA FILES 
        for x in range(1, int(tmpFile.shape[0]/img_size)): 
            for y in range(1, int(tmpFile.shape[1]/img_size)-1):
                crop_image = tmpFile[prev_x:x*img_size, y*img_size:(y*img_size + img_size)]
                
                crop_image = crop_image[..., numpy.newaxis]
                crop_image = crop_image.transpose(2, 0, 1)
                # TODO: Give value to new dimension?? 
               # print(crop_image.dtype)

                tensors.append(torch.from_numpy(crop_image))
            prev_x = x*img_size

        self.fits_file = tensors
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fits_file)

    def __getitem__(self, idx):
        sample = self.fits_file[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


##############################
# THIS CODE SHOWS 4 SAMPLES FROM THE DATASET
##############################

# fig = plt.figure() 

# for i in range(len(fits_dataset)):
# 	sample = fits_dataset[i]

# 	print(i, sample.shape)

# 	ax = plt.subplot(1, 4, i + 1)
# 	plt.tight_layout()
# 	ax.set_title('Sample #{}'.format(i))
# 	ax.axis('off')
# 	plt.imshow(sample, cmap='gray')
	
	

# 	if i == 3: 
# 		plt.show()
# 		break



################################
# THIS CODE SHOWS BATCHED SAMPLES FROM DATASET
# NOT CURRENTLY WORKING 
################################

# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch = sample_batched
#     batch_size = len(images_batch)
#     im_size = images_batch.size(1)

#     grid = utils.make_grid(images_batch)
#     # print(grid)
#     print(grid.shape)
#     plt.imshow(grid.numpy().transpose(1,2,0), cmap='gray')
#     plt.title('Batch from dataloader')

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched.size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break

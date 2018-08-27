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
from utils import progress_bar
import random
from regions import read_ds9, write_ds9

default_dimens = 276

class FitsHelper():
    def __init__(self, root_dir, dimensions=default_dimens):
        fitsFiles = os.listdir(root_dir)
        print("Reading fits files and storing dimensions for efficiency and logic... ")
        self.root_dir = root_dir 
        self.fitsFiles = fitsFiles 

        dim_arr = []
        for x in fitsFiles: 
            data = fits.getdata(root_dir + "/" + x)
            x = data.shape[0] - (data.shape[0] % dimensions)
            y = data.shape[1] - (data.shape[1] % dimensions)
            dim_arr.append((x,y))

        self.fits_dims = dim_arr 
        self.dimensions = dimensions
        print("Done.")

    def getFits(self):
        return self.fitsFiles 

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

class GalaxyHelper(): # TODO
    def __init__(self, fits_dir, galax_dir, nongalax_dir):

        self.galaxy_files = os.listdir(galax_dir)
        self.nongalax_files = os.listdir(nongalax_dir)
        self.fitsfiles = os.listdir(fitsdir)

        self.galax_dir = galax_dir
        self.fits_dir = fits_dir
        self.nongalax_dir = nongalax_dir

        self.nongalax_files = sorted(self.nongalax_files)
        self.galaxy_files = sorted(self.galaxy_files)
        self.fitsfiles = sorted(self.fitsfiles)

        # INDEX GOES GALAXIES THEN STARS FOR EACH FILE 
        self.pointCounts = []

        
        totalCount = 0 
        for x in range(len(self.fitsfiles)):
            # print(self.nongalax_files[x])
            # print(self.galaxy_files[x])
            galaxies = read_ds9(galax_dir + '/' + self.galaxy_files[x])      
            nongalaxies = read_ds9(nongalax_dir + '/' + self.nongalax_files[x])      


            self.pointCounts.append((totalCount, totalCount + len(galaxies), totalCount + len(galaxies) + len(nongalaxies))) # tuple with (0, galaxy length, totalLength)
            totalCount += len(galaxies) + len(nongalaxies)
            # print(totalCount)
        self.totalCount = totalCount
        print("Total objects", self.totalCount)

    def getDir(self):
        return self.fits_dir

    def getFits(self):
        return self.fitsfiles

    def getFileIndexes(self, idx): # parameter is file number
        return self.pointCounts[x]

    def getLength(self):
        return self.totalCount

    def getFileAndIdx(self, idx): # input is index and it returns file number and index
        count = 0
        totalCount = 0
        isGalaxy = False

        for x in range(len(self.pointCounts)):
            if (self.pointCounts[x][0] <= idx < self.pointCounts[x][2]):
                totalCount = self.pointCounts[x][0]
                break
            count += 1

        return (count, idx-totalCount) # returns file number and if it's galaxy or nongalaxy file

    def getCoordAndTarget(self, fileIdx, idx):
        objects = None
        target = 0
        if(idx < self.pointCounts[fileIdx][1]):
            objects = read_ds9(self.galax_dir + '/' + self.galaxy_files[fileIdx])      
        else:
            objects = read_ds9(self.nongalax_dir + '/' + self.nongalax_files[x])
            target = 1

        return (objects[idx], target)

class GalaxyDataset2(Dataset): # TODO 
    def __init__(self, galaxhelper, transform=None):
        self.galaxyHelp = galaxhelper
        self.transform = transform
        self.dimensions = 110

        self.curr_fits = 0
        self.data = fits.getdata(self.galaxyHelp.getDir() + "/" + self.galaxyHelp.getFits()[0])


    def __len__(self):
        self.galaxyHelp.getLength()

    def __getitem__(self, idx):
        fileNo, fidx = self.galaxyHelp.getFileAndIdx(idx)
        coord, target = self.galaxyHelp.getCoordAndTarget(fileNo, fidx)
        # coord.center.x 
        # coord.center.y

        x = int(round(coord.center.x)) - int(self.dimensions/2) 
        y = int(round(coord.center.y)) - int(self.dimensions/2) 

        if (self.curr_fits == fileNo):
            tmpFile = self.data
        else:
            tmpFile = self.galaxyHelp.getDir() + '/' + self.galaxyHelp.getFits()[fileNo]
            tmpFile = fits.getdata(tmpFile, ext=0) 
            self.curr_fits = fileNo
            self.data = tmpFile 

        crop_image = tmpFile[x:x+self.dimensions, y:y+self.dimensions]

        if self.transform:
            crop_image = self.transform(crop_image)


        # Converting to one channel tensor 
        crop_image = crop_image[..., numpy.newaxis]
        crop_image = crop_image.transpose(2, 0, 1)

        sample = torch.from_numpy(crop_image)

        savedir = 'newg/gal'
        if (target == 0):
            plt.imsave(savedir, sample, cmap='gray')

        return (crop_image.float(), target)


######################
# TEST
######################

galaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/galaxies'
nongalaxdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/nongalaxies'
fitsdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/fits'
transform_train = transforms.Compose([
    fits_loader.RandomCrop(96)
])



galaxhelper = GalaxyHelper(fits_dir=fitsdir, galax_dir=galaxdir, nongalax_dir=nongalaxdir) 
trainset = GalaxyDataset2(galaxhelper, transforms) 

for x in trainset:
    print('Shape: ' + str(x[0].shape) + ' type: ' + str(target))

######################
# TEST
######################


class GalaxyDataset(Dataset):
    # TODO: SO MUCH I/O going on, maybe do what FitsDataset does and workaround
    def __init__(self, root_dir, galaxies, transform=None):
               
        self.galaxies = pd.read_csv(galaxies, delim_whitespace=True)
        self.fits_files = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        count = 0

        galaxs = []
        blacklist = os.listdir('snapshots/galaxies/blacklist')
        galaxies = 0
        nongalax = 0

        # THE PURPOSE OF THIS IS TO GET RID OF BIGGER IMAGES 
        
        for x in range(len(self.galaxies)):
            for y in range(len(self.fits_files)):
                if self.galaxies.iloc[x, 0] in self.fits_files[y] and self.fits_files[y] not in blacklist: # index 0 is the ID of the file 
                    data = fits.getdata(self.root_dir + "/" + self.fits_files[y])

                    if(self.galaxies.iloc[x,15] == 0):
                        # print(str(count) + ' ' + str(data.shape) + ' ' + self.fits_files[y])
                        progress_bar(count, len(self.galaxies))
                        count += 1
                        targ = self.galaxies.iloc[x, 14]
                        galaxs.append((self.fits_files[y], data.shape, targ)) # index 14 is the class in the galaxies file  
                        if 0 < targ < 5:
                            galaxies += 1
                        else:
                            nongalax += 1
                    break
        
        print("Galaxies:", galaxies)
        print("Nongalax:", nongalax)
        print("Percentgalax:", galaxies/count*100)
        self.galaxies = galaxs 

    def __len__(self):
        return len(self.galaxies)

    def __getitem__(self, idx):
        target = 0
        if (6 <= self.galaxies[idx][2] <= 9):
            target = 1 

        # idx is index to a single galaxy fits image
        img = self.root_dir + '/' + self.galaxies[idx][0] 
        img = fits.getdata(img, ext=0) 

        d = 110
        x1 = int(round((self.galaxies[idx][1][0] - d) / 2.))
        y1 = int(round((self.galaxies[idx][1][1] - d) / 2.))
        img = img[x1:x1+d,y1:y1+d]

        if (target == 1):
            tmp = "snapshots/nongalax/" + str(self.galaxies[idx][0]) + ".png"
            plt.imsave(tmp, img, cmap='gray')

        if self.transform: # DONE: random crop images 256 * 256
            img = self.transform(img)
        
        # Converting to one channel tensor 
        img = img[..., numpy.newaxis]
        img = img.transpose(2, 0, 1)

        # RANDOM VERTICAL FLIP 
        if random.random() < 0.5:
            img[0] = numpy.flip(img[0], 1)

        # RANDOM HORIZONTAL FLIP
        if random.random() < 0.5:
            img[0] = numpy.flip(img[0], 0)

        # RANDOM ROTATE 
        n = random.choice([0, 1, 2, 3])
        img[0] = numpy.rot90(img[0], n)

        sample = torch.from_numpy(img)
        # RANDOM HORIZONTAL FLIP 
        

        # DONE: return a tuple (input, target) 
            # target is binary for now: galaxy or not galaxy 
        return (sample.float(), target)


class FitsDataset(Dataset):
    """Face Landmarks dataset."""
    # make this more efficient 
    # be able to know all files but also 
    # be able to index without reading every file in

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
        # print(image.shape)
        new_h, new_w = self.output_size

        top = numpy.random.randint(0, h - new_h)
        # print(w - new_w)
        # print('w: ', w)
        # print('new_w:', new_w)
        left = numpy.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        
        return image
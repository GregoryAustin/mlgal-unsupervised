import numpy as np
from astropy import wcs
from astropy.io import fits
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename

from regions import read_ds9, write_ds9
import os

import json

regdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/reg'
fitsdir = '/home/greg/Desktop/LabelledData/NN project/galaxnew/fits'

regfiles = os.listdir(regdir)
fitsfiles = os.listdir(fitsdir)

regfiles = sorted(regfiles)
fitsfiles = sorted(fitsfiles)

for x in range(len(fitsfiles)):
    data = fits.getdata(fitsdir + "/" + fitsfiles[x])
    print(fitsfiles[x])
    shape = data.shape
    print("shape: ", shape)
    tmp = regdir + '/' + regfiles[x]
    print(regfiles[x])
    regions = read_ds9(tmp, errors='warn')

    galaxObjs = []
    galaxies = 0
    nongalax = 0
    nongObjs = []

    # print(regions)

    for y in regions:
        if (0. < y.center.x < float(shape[0]) and 0. < y.center.y < float(shape[1])):
            # objects.append((y.center.x, y.center.y, y.meta["text"]))
            # objects.append(y)
            # print(str(x.meta["text"]))
            if (str(y.meta["text"]) == "{gal }"):
                galaxObjs.append(y)
                galaxies += 1
            else:
                nongObjs.append(y)
                nongalax += 1

            # print('(' + str(x.center.x) + ',' + str(x.center.y) + ') tag:' + str(x.meta["text"]))
    galaxFile = 'galax' + regfiles[x] + '.reg'
    nonGalaxFile = 'nong' + regfiles[x] + '.reg'

    # write_ds9(galaxObjs, galaxFile)
    # write_ds9(nongObjs,  nonGalaxFile)

    # print("Total    ", len(objects))
    print("Galaxies ", galaxies)
    print("Nongalax ", nongalax)
# for x in fitsFiles: 


# regions = read_ds9(file, errors='warn')
# print(len(regions))

# regions 


# def load_wcs_from_file(filename):
#     # Load the FITS hdulist using astropy.io.fits
#     hdulist = fits.open(filename)

#     hdulist.info()

# load_wcs_from_file('pptile1_K.fits')




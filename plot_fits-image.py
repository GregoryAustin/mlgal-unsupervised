# -*- coding: utf-8 -*-
"""
=======================================
Read and plot an image from a FITS file
=======================================

This example opens an image stored in a FITS file and displays it to the screen.

This example uses `astropy.utils.data` to download the file, `astropy.io.fits` to open
the file, and `matplotlib.pyplot` to display the image.

-------------------

*By: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz*

*License: BSD*

-------------------

"""

##############################################################################
# Set up matplotlib and use a nicer set of plot parameters
import os
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

##############################################################################
# Download the example FITS files used by this example:

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#fits_image_name = "/home/greg/Dropbox/Galaxy_Detection/w20050521_01263_sf_st_K.fit"

root_dir = '/home/greg/Desktop/Galaxyfits'
fitsFiles = os.listdir(root_dir)

for x in fitsFiles:
	data = fits.getdata(root_dir + "/" + x)
	print(data.shape)
	

##############################################################################
# Use `astropy.io.fits.info()` to display the structure of the file:

# image_file.info()


# ##############################################################################
# # Generally the image information is located in the Primary HDU, also known
# # as extension 0. Here, we use `astropy.io.fits.getdata()` to read the image
# # data from this first extension using the keyword argument ``ext=0``:

# image_data = fits.getdata(fits_image_name, ext=0)

# # ##############################################################################
# # # The data is now stored as a 2D numpy array. Print the dimensions using the
# # # shape attribute:

# print(image_data.shape)

# crop_image = image_data[500:1000, 2000:2500]

# print(crop_image.shape)

# # ##############################################################################
# # # Display the image data:

# plt.figure()
# plt.imshow(crop_image, cmap='gray')
# plt.colorbar()
# plt.show()
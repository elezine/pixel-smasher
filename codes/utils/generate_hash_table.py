# ek

##

# import matplotlib.pyplot as plt

import os, pickle
import numpy as np
#import rasterio as rio
#from rasterio import plot as rioplot
from xml.dom import minidom
# import matplotlib.colors as colors
import glob as gl
# from osgeo import gdal
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

##
files = gl.glob(r'/data_dir/planet_scenes/*.tif')
xmls = gl.glob(r'/data_dir/planet_scenes/*.xml')
hash=dict()
##

# n= 1 # testing
# with rio.open(files[n]) as src:
#     band_blue_radiance = src.read(1)
    
# with rio.open(files[n]) as src:
#     band_green_radiance = src.read(2)

# with rio.open(files[5]) as src:
#     band_red_radiance = src.read(3)

# with rio.open(files[n]) as src:
#     band_nir_radiance = src.read(4)

##
for n in range(len(files)):
    ID=os.path.basename(files[n])[:-4]
    xmldoc = minidom.parse(files[n][:-4]+'_metadata.xml')
    # xmldoc = minidom.parse(xmls[n])
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)

    print("Conversion coefficients for file {} - {} : {}".format(n, ID, coeffs))
    hash[ID]=coeffs

## save file
f=open("cal_hash.pkl", "wb")
pickle.dump(hash, f)
f.close()

## open file (test)
# f=open("cal_hash.pkl", "rb")
# hash=pickle.load(f)

## 
nodes[0].getElementsByTagName("ps:bandNumber")[0].firstChild.data
coeffs

##
# import xml.etree.ElementTree as ET
# import string
# xmlTree = ET.parse(xmls[0])
# tags = {elem.tag for elem in xmlTree.iter()}
# tags
# tags_list = list({elem.tag for elem in xmlTree.iter()})
# print(tags_list)


##
# root = xmlTree.getroot()
# for child in root:
#     print(child.tag, child.attrib)
# print(child.getchildren)


# ## Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients

# ##
# # Set spatial characteristics of the output object to mirror the input
# kwargs = src.meta
# kwargs.update(
#     dtype=rio.uint16,
#     count = 4)

# print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))

# # Here we include a fixed scaling factor. This is common practice.
# scale = 10000
# blue_ref_scaled = scale * band_blue_reflectance
# green_ref_scaled = scale * band_green_reflectance
# red_ref_scaled = scale * band_red_reflectance
# nir_ref_scaled = scale * band_nir_reflectance

# print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled), np.amax(red_ref_scaled)))

# # Write band calculations to a new raster file
# with rio.open('reflectance.tif', 'w', **kwargs) as dst:
#         dst.write_band(1, band_blue_reflectance.astype(rio.uint16))
#         dst.write_band(2, band_green_reflectance.astype(rio.uint16))
#         dst.write_band(3, band_red_reflectance.astype(rio.uint16))
#         dst.write_band(4, band_nir_reflectance.astype(rio.uint16))

# ##
# """
# The reflectance values will range from 0 to 1. You want to use a diverging color scheme to visualize the data,
# and you want to center the colorbar at a defined midpoint. The class below allows you to normalize the colorbar.
# """

# class MidpointNormalize(colors.Normalize):
#     """
#     Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
#     e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
#     Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
#     """
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# # Set min/max values from reflectance range for image (excluding NAN)
# min = np.nanmin(band_nir_reflectance)
# max = np.nanmax(band_nir_reflectance)
# mid = 0.20

# fig = plt.figure(figsize=(20,10))
# ax = fig.add_subplot(111)

# # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
# # note that appending '_r' to the color scheme name reverses it!
# cmap = plt.cm.get_cmap('RdGy_r')

# cax = ax.imshow(band_nir_reflectance, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))

# ax.axis('off')
# ax.set_title('NIR Reflectance', fontsize=18, fontweight='bold')

# cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)

# plt.show()

# ##

# ##

# ##

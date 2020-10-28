import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
import matplotlib.pyplot as plt
from water_mask_funcs import create_buffer_mask
import os
import rasterio as rio

HR_img = cv2.imread('/data_dir/hold_mod_scenes-shield-gt-subsets/HR/x10/20170710_181144_1034_3B_AnalyticMS_SR_s0001.png', cv2.IMREAD_UNCHANGED)
HR_mask = cv2.imread('/data_dir/hold_mod_scenes-shield-gt-subsets_masks/HR/x10/20170710_181144_1034_3B_AnalyticMS_SR_s0001_no_buffer_mask.png', cv2.IMREAD_UNCHANGED)

HR_img = np.single(HR_img)

ndwi_bands = (2,1)
foreground_threshold = 127
buffer_additional = 0

buffer_mask = create_buffer_mask(HR_mask, foreground_threshold, buffer_additional)

labeled = measure.label(buffer_mask, background=0, connectivity=2)

bw = np.full(labeled.shape, False)
copy = np.full(labeled.shape, False)
regions = measure.regionprops(labeled)
water_index = HR_img[:,:,ndwi_bands[0]]

class compare:
    def __init__(self, a):
        self.a = a

    # reverse greater than symbol!
    def __gt__(self, o):
        return self.a <= o.a 
    

for x,region in enumerate(regions):
    print('region: ' + str(x))
    print('before changing, val 0,0 is ' + str(copy[0,0]))
    
    coords = region.coords
    i = coords[:,0]
    j = coords[:,1]

    print('the min i coord is: ' + str(min(i)))
    print('the min j coord is: ' + str(min(j)))
    
    dist=0 # being lazy and modified from create_buffer_mask_fxn
    bbox_coords = region.bbox #(min_row, min_col, max_row, max_col)
    if bbox_coords[0] - dist >= 0:
        bbox_i_min = bbox_coords[0] - dist
    else: 
        bbox_i_min = 0
    if bbox_coords[1] - dist >= 0:
        bbox_j_min = bbox_coords[1] - dist
    else:
        bbox_j_min = 0
    if bbox_coords[2] + dist <= copy.shape[0]:
        bbox_i_max = bbox_coords[2] + dist
    else:
        bbox_i_max = copy.shape[0]
    if bbox_coords[3] + dist <= copy.shape[1]:
        bbox_j_max = bbox_coords[3] + dist
    else:
        bbox_j_max = copy.shape[1]

    copy_x = copy[i,j]
    ndwi_x = water_index[i,j]
    thresh_x = threshold_otsu(ndwi_x)
    copy_x = compare(ndwi_x)>compare(thresh_x)
    copy[i,j] = copy_x
    
    print('bbox_i_min :' + str(bbox_i_min))
    print('bbox_i_max :' + str(bbox_i_max))
    print('bbox_j_min :' + str(bbox_j_min))
    print('bbox_j_max :' + str(bbox_j_max))
    
    print('after changing, val 0,0 is ' + str(copy[0,0]))
    print('--')
    

copy[copy == False] = 0
copy[copy == True] = 1

georef_path = '/data_dir/Scenes-shield-gt-subsets/20170710_181144_1034_3B_AnalyticMS_SR_s0001.tif'

georef_name = os.path.basename(georef_path)

non_georef_img = copy

georef_img_rio = rio.open(georef_path)
profile = georef_img_rio.profile
profile.update(nodata = 255)

dtype = str(non_georef_img.dtype)
if dtype == 'uint8':
    dtype = rio.uint8
if dtype == 'float64':
    dtype = rio.float64
if dtype == 'bool':
    dtype = rio.bool
if dtype == 'uint16':
    dtype = rio.uint16
if dtype == 'int16':
    dtype = rio.int16
if dtype == 'uint32':
    dtype = rio.uint32
if dtype == 'int32':
    dtype = rio.int32
if dtype == 'float32':
    dtype = rio.float32

if len(np.shape(non_georef_img)) > 2:
    profile.update(dtype = dtype, count = 3)

    new_non_georef_img = np.zeros((non_georef_img.shape[0], non_georef_img.shape[1], 3))
    new_non_georef_img.fill(255)
    new_non_georef_img[:,:,:] = non_georef_img[:,:,:]

else:
    profile.update(dtype = dtype, count = 1)

    #non_georef_img[non_georef_img < 1] = 0
    #non_georef_img[non_georef_img > 1] = 1

    new_non_georef_img = np.zeros((non_georef_img.shape[0], non_georef_img.shape[1], 1))
    new_non_georef_img.fill(255)
    new_non_georef_img[:,:,0] = non_georef_img[:,:]

with rio.Env():
    with rio.open('ex.tif', 'w', **profile) as dst:
        new_non_georef_img = np.rollaxis(new_non_georef_img, 2)
        dst.write(new_non_georef_img.astype(dtype))
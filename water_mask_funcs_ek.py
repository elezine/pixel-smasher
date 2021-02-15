from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, selem
import numpy as np
import matplotlib.pyplot as plt
# import rasterio as rio
import os
import glob as gl
from osgeo import gdal,ogr,osr,gdalconst

# I/O:
dilation_radius_step_sz=25 # largest possible dilation for one step

def create_buffer_mask(og_mask, foreground_threshold=0, buffer_additional=0):
    '''
    Function to add an equal-area buffer to regions in a binary image. Foreground threshold is used if input binary image had been resampled with bicubic or quadratic and thus has multiple greyscale levels. 127 would be a reasonable number for foreground_threshold in this case.
    Inputs:
        OG_mask                 binary mask with background = 0
        foreground_threshold    maximum value to treat as background (rest become foreground)
        buffer_additional       additional pixels to buffer (+/-) to modify equal-area default
    '''
    
    og_mask[og_mask > foreground_threshold] = 1
    labeled = measure.label(og_mask, background=0, connectivity=2)
    if np.all(og_mask==1):
        mask = np.full(labeled.shape, True)
    elif np.all(og_mask==0):
        mask = np.full(labeled.shape, False)
    else:
        regions = measure.regionprops(labeled)
        mask = np.full(labeled.shape, False)
        copy = np.full(labeled.shape, False)
        for x,region in enumerate(regions):
            old_radius = region.equivalent_diameter/2
            old_area = np.pi*old_radius**2
            new_area = 2*old_area
            new_radius = np.sqrt(new_area/np.pi)

            buffer = new_radius-old_radius + buffer_additional

            coords = region.coords
            i = coords[:,0]
            j = coords[:,1]

            copy[i,j] = True

            dist = int(np.ceil(buffer))

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

            #for i in range(0, int(np.ceil(buffer))):
            copy_x = copy[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max]
            if dist > dilation_radius_step_sz*2:
                dist_tmp = dilation_radius_step_sz
                dist_remain = dist # init
                while (dist_tmp > 0) & (dist_remain > 0):
                    strelem = selem.disk(dist_tmp)
                    copy_x = binary_dilation(copy_x, selem = strelem)
                    dist_remain=dist_remain-dist_tmp # after this iter
                    dist_tmp = np.min([dist_remain-dist_tmp, dilation_radius_step_sz]) # for next iter

            else:
                # strelem = np.ones((dist*2+1,dist*2+1)) # box
                strelem = selem.disk(dist) # disk
                copy_x = binary_dilation(copy_x, selem = strelem)
            copy[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max] = copy_x
            #bounds = find_boundaries(pekel_copy)
            #pekel_copy[bounds] = 1

            mask = mask | copy
    
    mask[mask>0] = 1
    
    return mask
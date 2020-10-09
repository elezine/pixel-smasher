from skimage import measure
#from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, selem
import numpy as np
#import matplotlib.pyplot as plt
#import rasterio as rio
#import os
#import glob as gl
#from osgeo import gdal,ogr,osr,gdalconst

# I/O:
dilation_radius_step_sz=25
'''

def reproj_ras_to_ref(raster_path, reference_path, output_path):

    input = gdal.Open(raster_path, gdalconst.GA_ReadOnly)
    inputProj = input.GetProjection()
    inputTrans = input.GetGeoTransform()
    bandreference_0 = input.GetRasterBand(1)

    reference = gdal.Open(reference_path, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference_1 = reference.GetRasterBand(1)
    x = reference.RasterXSize
    y = reference.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(output_path, x, y, 1, bandreference_0.DataType)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(input, output, inputProj, referenceProj, gdalconst.GDT_Float32)
    del output

    print('done reprojecting raster')
    
def buffer_mask(og_mask_path, ref_path, output_mask_path_buffer, output_mask_path_no_buffer):
    mask_reproj = r'mask_reproj.tif'
    reproj_ras_to_ref(og_mask_path, ref_path, mask_reproj)
    
    og_mask = rio.open(mask_reproj).read(1)
    ref_nan = rio.open(ref_path).read(1).astype(float)
    ref_nan[ref_nan == 0] = np.nan
    ref_nan[ref_nan > 0 ] = 1
    
    ref_int = np.nan_to_num(ref_nan, nan = 0).astype(int)
    
    og_mask = og_mask*ref_nan
    og_mask[og_mask>0] = 1

    #
    labeled = measure.label(og_mask, background=0, connectivity=2)
    labeled = labeled*ref_int
    
    labeled_split = np.array_split(labeled, 15)
    mask_split = []
    
    for l in labeled_split:
        regions = measure.regionprops(l)
        mask = np.full(l.shape, False)

        for x,region in enumerate(regions):

            copy = np.full(l.shape, False)

            old_radius = region.equivalent_diameter/2
            old_area = np.pi*old_radius**2
            new_area = 2*old_area
            new_radius = np.sqrt(new_area/np.pi)

            buffer = new_radius-old_radius

            coords = region.coords
            i = coords[:,0]
            j = coords[:,1]

            copy[i,j] = True

            dist = int(np.ceil(buffer))

            selem = np.ones((dist*2+1,dist*2+1))

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

            copy_x = binary_dilation(copy_x, selem = selem)
            copy[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max] = copy_x
                #bounds = find_boundaries(pekel_copy)
                #pekel_copy[bounds] = 1

            mask = mask + copy
            
        mask_split.append(mask)
    
    mask = np.concatenate(mask_split)
    mask[mask>0] = 1
    mask = mask.astype(float)*ref_nan
    #
    
    with rio.Env():
        with rio.open(ref_path) as src:
            profile = src.profile

            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                dtype= rio.uint8,
                count=1)

            #with rio.open(output_mask_path_buffer, 'w', **profile) as dst:
            #    dst.write(mask.astype(rio.uint8), 1)
            
            with rio.open(output_mask_path_no_buffer, 'w', **profile) as dst:
                dst.write(og_mask.astype(rio.uint8), 1)
                
                
#files = gl.glob('/data_dir/Scenes-shield/*.tif')
#for file in files:
#    name = (os.path.basename(os.path.normpath(file)))
#    output_name_buffer = '/data_dir/Water_mask/' + str(name) + '_buffer_mask.tif'
#    output_name_no_buffer =  '/data_dir/Water_mask/' + str(name) + '_no_buffer_mask.tif'
#    buffer_mask('/data_dir/pixelsmasher/pekel.tif', file, output_name_buffer, output_name_no_buffer)

'''
    
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
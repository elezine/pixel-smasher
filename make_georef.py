import numpy as np
from multiprocessing import Pool
import multiprocessing
import rasterio as rio
import glob as gl
import cv2
import os
import sys

non_georef_folder = '/data_dir/hold_mod_scenes-shield-gt-subsets/HR/x10/'

#data_dir/classified_shield_v2/008_ESRGAN_x10_PLANET_noPreTrain_130k_Shorelines_Test/visualization/Bic/x10/20170707_181137_1035_3B_AnalyticMS_SR_s0001_T0.png

non_georef_suffix = '' #'008_ESRGAN_x10_PLANET_noPreTrain_130k_Test'

georef_folder = '/data_dir/Scenes-shield-gt-subsets/'
#data_dir/Scenes-shield-gt-subsets/20170707_181137_1035_3B_AnalyticMS_SR_s0001.tif

save_folder = '/data_dir/classified_shield_gt_subsets_georef/unclassified_test/'

img_list = gl.glob(georef_folder + '*.tif')

def main():
    """A multi-thread tool to georef images."""

    #n_thread = multiprocessing.cpu_count() #1
    
    #pool = Pool(4) # (n_thread)
    #for path in img_list:
    #    pool.apply_async(worker,
                         #args=(path, SR_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz))
    #pool.close()
    #pool.join()
    #print('All subprocesses done.')
    
    for path in img_list:
        worker(non_georef_folder, path, non_georef_suffix, save_folder)

def worker(non_georef_folder, georef_path, non_georef_suffix, save_folder): 
    '''
    This worker works on one non georef image at a time to make it georeferenced.

    '''
    # load georeffed image:
    georef_name = os.path.basename(georef_path)

    georef_img_rio = rio.open(georef_path)
    profile = georef_img_rio.profile
    profile.update(nodata = 255)
    
    #load nongeoreffed image:
    non_georef_path = str(non_georef_folder) + georef_name.replace('.tif', str(non_georef_suffix)) + '.png'
    non_georef_img = cv2.imread(non_georef_path, cv2.IMREAD_UNCHANGED)
    
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
    
        non_georef_img[non_georef_img < 1] = 0
        non_georef_img[non_georef_img > 1] = 1
    
        new_non_georef_img = np.zeros((non_georef_img.shape[0], non_georef_img.shape[1], 1))
        new_non_georef_img.fill(255)
        new_non_georef_img[:,:,0] = non_georef_img[:,:]

    save_path = save_folder + georef_name
    
    with rio.Env():
        with rio.open(save_path, 'w', **profile) as dst:
            new_non_georef_img = np.rollaxis(new_non_georef_img, 2)
            dst.write(new_non_georef_img.astype(dtype))
    
    return 'Processing {:s} ...'.format(georef_name)
    
if __name__ == '__main__':
    main()
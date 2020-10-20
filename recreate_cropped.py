import numpy as np
from multiprocessing import Pool
import multiprocessing
import rasterio as rio
import glob as gl
import cv2
import os
import sys

cropped_folder = '/data_dir/classified_shield_v2/008_ESRGAN_x10_PLANET_noPreTrain_130k_Test_hold_shield_v2/visualization/HR/x10/'
cropped_suffix = 'T0' #'008_ESRGAN_x10_PLANET_noPreTrain_130k_Test'

uncropped_folder = '/data_dir/Scenes-shield-gt/'

save_folder = '/data_dir/classified_shield_gt_georef/HR/10x/'

img_list = gl.glob(uncropped_folder + '*SR.tif')

def main():
    """A multi-thread tool to put cropped images back into full (non-cropped) array based on index."""

    #n_thread = multiprocessing.cpu_count() #1
    
    crop_sz = 480 # num px in x and y
    step = 240
    thres_sz = 48
    
    #pool = Pool(4) # (n_thread)
    #for path in img_list:
    #    pool.apply_async(worker,
                         #args=(path, SR_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz))
    #pool.close()
    #pool.join()
    #print('All subprocesses done.')
    
    for path in img_list:
        worker(path, cropped_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz)

def worker(uncropped_path, cropped_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz): 
    '''
    This worker works on one uncropped image at a time to get the correct indexes for each cropped subset of the uncropped image,
    grab that cropped subset, and put it into a new array.

    '''
    # load uncropped image:
    uncropped_name = os.path.basename(uncropped_path)
    uncropped_img = cv2.imread(uncropped_path, cv2.IMREAD_UNCHANGED)
    
    uncropped_img_rio = rio.open(uncropped_path)
    profile = uncropped_img_rio.profile
    profile.update(nodata = 255)
    
    #get output size:
    cropped_path = gl.glob(cropped_folder + '*.png')[0]
    cropped_ex = cv2.imread(cropped_path, cv2.IMREAD_UNCHANGED)
    
    dtype = str(cropped_ex.dtype)
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
    
    n_channels = len(cropped_ex.shape)
    if n_channels == 2:
        h, w = uncropped_img.shape[0:2]
        new_uncropped_img = np.zeros((h,w,1))
        new_uncropped_img.fill(255)
        profile.update(dtype = dtype, count = 1)
    elif n_channels == 3:
        h, w, c = uncropped_img.shape
        new_uncropped_img = np.zeros((h,w,3))
        new_uncropped_img.fill(255)
        profile.update(dtype = dtype, count = 3)
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)
    
    index = 1
    for x in h_space:
        for y in w_space:
            if n_channels == 2:
                crop_img = uncropped_img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = uncropped_img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            if ~np.any(np.sum(crop_img,axis=2)==0): # if all three bands == 0
                
                cropped_path = str(cropped_folder) + uncropped_name.replace('.tif', '_s{:04d}_'.format(index)) + str(cropped_suffix) + '.png'
                #print(cropped_path)
                #print(os.path.isfile(cropped_path))
                
                try:
                    cropped_img = cv2.imread(cropped_path, cv2.IMREAD_UNCHANGED)
                    if len(np.unique(cropped_img)) == 2:
                        cropped_img[cropped_img < 1] = 0
                        cropped_img[cropped_img > 1] = 1
                except:
                    cropped_img = np.zeros(crop_img.shape)
                    cropped_img.fill(255)
                
                if n_channels == 2:
                    new_uncropped_img[x:x + crop_sz, y:y + crop_sz, 0] = cropped_img[:]
                else:
                    new_uncropped_img[x:x + crop_sz, y:y + crop_sz, :] = cropped_img[:]
                
                index += 1
            else:
                #print('all zero')
                pass
    
    save_path = save_folder + uncropped_name
    
    with rio.Env():
        with rio.open(save_path, 'w', **profile) as dst:
            new_uncropped_img = np.rollaxis(new_uncropped_img, 2)
            dst.write(new_uncropped_img.astype(dtype))
    
    return 'Processing {:s} ...'.format(uncropped_name)
    
    
if __name__ == '__main__':
    main()

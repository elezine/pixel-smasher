import numpy as np
from multiprocessing import Pool
import multiprocessing
import rasterio as rio
import glob as gl
import cv2
import os
import sys

cropped_suffix = '008_ESRGAN_x10_PLANET_noPreTrain_130k_Test'
save_folder = '/data_dir/SR_georef_test/'
SR_folder = '/data_dir/pixel-smasher/results/008_ESRGAN_x10_PLANET_noPreTrain_130k_Test/visualization/hold_mod_shield_v2/'
HR_folder = '/data_dir/Scenes-shield-gt/'

img_list = gl.glob(HR_folder + '*SR.tif')

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
        worker(path, SR_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz)

def worker(HR_path, cropped_SR_folder, cropped_suffix, save_folder, crop_sz, step, thres_sz): 
    '''
    This worker works on one HR image at a time to get the correct indexes for each cropped SR subset of the big HR image,
    grab that cropped image, and put it into a new array.

    '''
    # load big image (HR):
    HR_name = os.path.basename(HR_path)
    #print('HR path: ' + HR_path)
    HR_img = cv2.imread(HR_path, cv2.IMREAD_UNCHANGED)
    HR_img_rio = rio.open(HR_path)
    
    new_img = np.zeros((HR_img.shape[0], HR_img.shape[1], 3))
    
    n_channels = len(HR_img.shape)
    if n_channels == 2:
        h, w = HR_img.shape
    elif n_channels == 3:
        h, w, c = HR_img.shape
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
        #print(x)
        for y in w_space:
            #print(y)
            if n_channels == 2:
                crop_img = HR_img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = HR_img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            if ~np.any(np.sum(crop_img,axis=2)==0): # if all three bands == 0
                
                SR_path = str(cropped_SR_folder) + HR_name.replace('.tif', '_s{:04d}_'.format(index)) + str(cropped_suffix) + '.png'
                #print('SR path is : ' + SR_path)
                
                try:
                    SR_image = cv2.imread(SR_path, cv2.IMREAD_UNCHANGED)
                    #print('SR image is shape : ' + str(np.shape(SR_image)))
                    #print(os.path.exists(SR_path))
                except:
                    SR_image = np.zeros(crop_img.shape)
                
                if n_channels == 2:
                    new_img[x:x + crop_sz, y:y + crop_sz] = SR_image[:]
                else:
                    new_img[x:x + crop_sz, y:y + crop_sz, :] = SR_image[:]
                
                index += 1
            else:
                #print('all zero')
                pass
    
    save_path = save_folder + HR_name
    profile = HR_img_rio.profile

    with rio.Env():
        with rio.open(save_path, 'w', **profile) as dst:
            dst.write(new_img)
    
    return 'Processing {:s} ...'.format(HR_name)
    
    
if __name__ == '__main__':
    main()

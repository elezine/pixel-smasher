import os
import os.path as osp
import sys
import getpass
from multiprocessing import Pool
import multiprocessing
import numpy as np
import cv2
from skimage import exposure
from skimage.util import img_as_ubyte

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError as e:
    print('Error caught: '+str(e))
    pass

btm_percentile=2
top_percentile=99
band_order=(2,1,3)  # 3,2,1 for NRG, 2,1,3 for RGN, 2,1,0 for RGB (original = BGRN)
ndwi_bands=(1,3) # (1,3) # used to determine maximum or (n-percentile) brightness in scene

def main():
    """A multi-thread tool to crop sub imags."""
    if getpass.getuser()=='ekyzivat': # on ethan local
        input_folder = 'F:\ComputerVision\Planet'
        save_folder = 'F:\ComputerVision\Planet_sub'
    elif getpass.getuser()=='ethan_kyzivat' or getpass.getuser()=='ekaterina_lezine': # on GCP 
        input_folder = '/data_dir/Scenes'
        save_folder = '/data_dir/planet_sub'
    else: # other
        raise ValueError('input_folder not specified!')
        pass

    n_thread = n_thread = multiprocessing.cpu_count() #1
    crop_sz = 480 # num px in x and y
    step = 240
    thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        # print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        # sys.exit(1)
        pass # uncomment above two lines for ease of working, if necessary

    img_list = []
    for root, dirsfoo, file_list in sorted(os.walk(input_folder)): # +'/*SR.tif'
        for x in file_list:  # assume only images in the input_folder
            if x.endswith("SR.tif"):
                path = os.path.join(root, x) 
                img_list.append(path)
        break
    # img_list = ['/data_dir/Scenes/20190619_191648_25_106f_3B_AnalyticMS_SR.tif'] # for testing
    def update(arg):
        pbar.update(arg)
    # img_list=img_list[:9] # for testing
    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

def rescale_reflectance_equal(img, reflectance_upper=10000):
        # for reflectance scaling
    reflectance_lower=0 # if setting to >0, need to rewrite function to include this as an input and also note that NDWI will not be constant

        ## apply reflectance scaling correction
    image_cal=np.ones(img.shape, dtype='single')
    # ID=filename[:-10]
    # coeffs=hash[ID]
    image_cal=np.minimum((img.astype(np.single)-reflectance_lower)/(reflectance_upper-reflectance_lower), image_cal) ## need to modify if refl_lower is > 0...
    # np.minimum is used to ensure that image value is clipped at one -> 255
    # image=cv2.normalize(image_cal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # this method uses min of empircal data as original min bound, so is not uniform between images
    img_rescaled=(image_cal*255).astype(np.uint8)
    return img_rescaled

def rescale_reflectance(img, btm_percentile=2, top_percentile=98):
        # Contrast stretching
    for i in range(img.shape[2]):
        btm_val = np.percentile(img[:,:,i][img[:,:,i]>0], btm_percentile)
        top_val = np.percentile(img[:,:,i][img[:,:,i]>0], top_percentile)
        img[:,:,i] = exposure.rescale_intensity(img[:,:,i], in_range=(btm_val, top_val))
    img=img_as_ubyte(img) #(img/65535*255).astype(np.uint8) # 
    return img

def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    reflectance_lower=np.percentile(img[:,:,ndwi_bands][img[:,:,ndwi_bands]>0], btm_percentile) # Compute maximum reflectance from entire 
    reflectance_upper=np.percentile(img[:,:,ndwi_bands][img[:,:,ndwi_bands]>0], top_percentile) # Compute maximum reflectance from entire scene, not individual subsets
    print(f'\n\nLoaded image:\t{img_name}')
    print(f'Rescaling reflectance to: {reflectance_lower:.1f} - {reflectance_upper:.1f} ish\n')

       # rescale and overwrite to img
    img=rescale_reflectance(img[:,:,band_order], btm_percentile, top_percentile)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
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
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            if ~np.any(np.sum(crop_img,axis=2)==0): # if all three bands == 0

                    # write
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.tif', '_s{:04d}.png'.format(index))),
                    crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                print(f'\t{img_name}\tCropped: {x}, {y}.')
                index += 1
            else:
                # print('\tSome No Data pixels in: {:d}, {:d}.  Skipping.'.format(x,y))
                pass

    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()

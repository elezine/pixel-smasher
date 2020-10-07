'''
Preprocessing: run 3rd
'''
import os
import os.path as osp
import sys
import getpass
from multiprocessing import Pool
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import exposure
from skimage.util import img_as_ubyte

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError as e:
    print('Error caught: '+str(e))
    pass

    # for relative stretch
btm_percentile=1
top_percentile=95
ndwi_bands=(3,1) # (1,3) # used to determine maximum or (n-percentile) brightness in scene (N,G)- not important for writing

    # for abs stretch
reflectance_upper=None #3000 # only used if using function 'rescale_reflectance_equal'
band_order=(0,1,3)  # (2,1,3) # If no opencv reversals: 3,2,1 for NRG, 2,1,3 for RGN, 2,1,0 for RGB (original = BGRN)
# if opencv reversal with write compensation (sees RGBN, then don't reverse bc it writes in reverse, but I flip output): 1,0,3 for NRG, 3,1,0 for RGN; 2,1,0 for RGB
# with opencv reversal (sees RGBN, then reverse bc it writes in revers): 3,0,1 for NRG, 0,1,3 for RGN; 0,1,2 for RGB
# opencv load BGRN as RGBN!

    # folder I/O
input_folder = '/data_dir/Scenes' # '/data_dir/Scenes-shield'
save_folder = '/data_dir/planet_sub_v2' # /data_dir/planet_sub/hold_mod_shield_v2.2 # planet_sub/hold_mod_shield_v2 is for individ image rescaling, not global
input_mask_folder = None # '/data_dir/Shield_Water_Mask' # None #'/data_dir/Shield_Water_Mask' # set to None if not using masks
save_mask_folder = '/data_dir/planet_sub_masks_v2' # /data_dir/planet_sub/hold_mod_shield_masks
save_hist_plot_folder = '/data_dir/other/hists/hists_hold_mod_v2' # set to None to not save or plot

    # parma I/O
n_thread = multiprocessing.cpu_count() #1 # 2
crop_sz = 480 # num px in x and y
step = 240
thres_sz = 48
compression_level = 3  # 3 is the default value in cv2
def main():
    """A multi-thread tool to crop sub imags."""
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

        # create output folders if they don't already exist
    for dir in [save_folder, save_mask_folder,save_hist_plot_folder]:
        if dir != None:
            if not os.path.exists(dir):
                os.makedirs(dir)
                print('mkdir [{:s}] ...'.format(dir))

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
    # img_list=img_list[238:270] # for testing
    pbar = ProgressBar(len(img_list))
    pool = Pool(n_thread) # (n_thread)
    for path in img_list:
        if input_mask_folder==None:
            path_mask=None
        else:
            path_mask=name_lookup(path) # lookup mask path
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level, path_mask, save_mask_folder),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

def rescale_reflectance_equal(img, reflectance_upper=10000):
    '''Rescales each band the same way, given an upper reflectance. Lower reflectance is always zero- so transformation is rescaling only, no offset. Preserves input nodata value of 0.'''
        # for reflectance scaling
    reflectance_lower=0 # if setting to >0, need to rewrite function to include this as an input and also note that NDWI will not be constant

    mask=np.sum(img,axis=2)==0

        ## apply reflectance scaling correction
    image_cal=np.ones(img.shape, dtype='single')
    # ID=filename[:-10]
    # coeffs=hash[ID]
    image_cal=np.minimum((img.astype(np.single)-reflectance_lower)/(reflectance_upper-reflectance_lower), image_cal) ## need to modify if refl_lower is > 0...
    # np.minimum is used to ensure that image value is clipped at one -> 255
    # image=cv2.normalize(image_cal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # this method uses min of empircal data as original min bound, so is not uniform between images
    img_rescaled=(image_cal*255).astype(np.uint8)

            # preserve nodata value
    img_rescaled[img_rescaled<255]+=1
    img_rescaled[mask]=0 # set nodata==0
    return img_rescaled

def rescale_reflectance(img, btm_percentile=2, top_percentile=98, individual_band=True):
    '''Rescales each band in each image individualy (unless flag 'individual_band' is False), given input btm and top percentiles. Preserves input nodata value of 0.'''
        # Contrast stretching
    mask=np.sum(img,axis=2)==0
    btm_val, top_val=[None]*img.shape[2], [None]*img.shape[2] # init
    if individual_band==True:
        for i in range(img.shape[2]):
            btm_val[i] = np.percentile(img[:,:,i][img[:,:,i]>0], btm_percentile)
            top_val[i] = np.percentile(img[:,:,i][img[:,:,i]>0], top_percentile)
            img[:,:,i] = exposure.rescale_intensity(img[:,:,i], in_range=(btm_val[i], top_val[i]))
    else:
        btm_val = np.percentile(img[np.sum(img, 2)>0], btm_percentile)
        top_val = np.percentile(img[np.sum(img, 2)>0], top_percentile)
        img = exposure.rescale_intensity(img, in_range=(btm_val, top_val))
    img=img_as_ubyte(img) #(img/65535*255).astype(np.uint8) # 

            # preserve nodata value
    img[img<255]+=1
    img[mask]=0 # set nodata==0
    return img

def rescale_reflectance_equal_per_band(img, limits):
    '''Rescales each band given in the input matrix, 'limits'
    limits: top row: btm limits, bottom row: top limits. Columns=bands 
    Sets (0,0,0) to 0 with no rescaling. Preserves input nodata value of 0.'''
        # Contrast stretching
    mask=np.sum(img,axis=2)==0 # nodata mask
    for i in range(img.shape[2]):
        img[:,:,i] = exposure.rescale_intensity(img[:,:,i], in_range=(limits[0,i], limits[1,i])) # TODO: vectorize this part! Maybe need to use my own rescaling function...
    # (img/65535*255).astype(np.uint8) # 

        # preserve nodata value
    img=img_as_ubyte(img)
    img[img<255]+=1
    img[mask]=0 # set nodata==0
    return img

def name_lookup(name_scene):
    '''
    Uses global var 'input_mask_folder'
    '''
    name_mask_scene = name_scene.replace(input_folder , input_mask_folder ).replace('.tif', '_no_buffer_mask.tif')
    return name_mask_scene

def worker(path, save_folder, crop_sz, step, thres_sz, compression_level, path_mask=None, save_mask_folder=None): # HERE TODO: load matrix
    '''
    input: pixel-smasher/quantile_matrix.npy
    '''
        # load 
    quantile_val=np.load('/home/ethan_kyzivat/code/pixel-smasher/quantile_matrix.npy')
    print(f'Loaded quantiles values:\n{quantile_val}')
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if np.any(img==None):
        raise ValueError(f'image {path} not found...')
    else:
        print(f'\n\nLoaded image :\t{img_name}')
    if path_mask != None:
        mask_name = os.path.basename(path_mask)
        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)
        print(f'\n\nLoaded mask:\t{mask_name}')
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError('Image and mask are different shapes.')
        # for relative stretch 
    # reflectance_lower=np.percentile(img[:,:,ndwi_bands][img[:,:,ndwi_bands]>0], btm_percentile) # Compute maximum reflectance from entire 
    # reflectance_upper=np.percentile(img[:,:,ndwi_bands][img[:,:,ndwi_bands]>0], top_percentile) # Compute maximum reflectance from entire scene, not individual subsets

       # rescale and overwrite to img : for relative stretch
    # img=rescale_reflectance(img[:,:,band_order], btm_percentile, top_percentile)
    # print(f'Rescaling reflectance to: {reflectance_lower:.1f} - {reflectance_upper:.1f} ish\n')
    
       # rescale and overwrite to img : for abs stretch
    # print(f'Rescaling reflectance to: {reflectance_upper:.1f}\n')
    print(f'Rescaling reflectance...')
    img=rescale_reflectance(img[:,:,band_order], btm_percentile, top_percentile, individual_band=False) # rescale_reflectance_equal_per_band(img[:,:,band_order], quantile_val[:, band_order])
    
        # save hist plots: view in NGR as written to file, not as opencv sees it (RGN)
    if save_hist_plot_folder != None:
        print('Calculating histogram...')
        f, ax = plt.subplots(img.shape[2], 2, sharex=True)
        for i in range(img.shape[2]):
            ax[i,1].hist(img[:,:,2-i][img[:,:,2-i]>0].flatten(),bins=np.linspace(0,255,256))
            ax[i,1].set_title('Write band: {}'.format(i))
        f.add_subplot(1,2,1)
        ax[0,0].set_title(img_name)
        plt.imshow(img[:,:,[2,1,0]], resample=True)
        plt.savefig(os.path.join(save_hist_plot_folder, img_name.replace('.tif', '_hist.png')))
        plt.close()
        print(f'\t{img_name} hist\tSaved.')

        # crops
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
                if path_mask != None:
                    crop_mask_img = mask[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                if path_mask != None:
                    crop_mask_img = mask[x:x + crop_sz, y:y + crop_sz] # samesies
            crop_img = np.ascontiguousarray(crop_img)
            if path_mask != None:
                crop_mask_img = np.ascontiguousarray(crop_mask_img)*255 # for vis purposes
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            if ~np.any(np.sum(crop_img,axis=2)==0): # if all three bands == 0

                    # write
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.tif', '_s{:04d}.png'.format(index))),
                    crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                print(f'\t{img_name}\tCropped: {x}, {y}.')
                if path_mask != None:
                    cv2.imwrite(
                        os.path.join(save_mask_folder, mask_name.replace('.tif', '_s{:04d}.png'.format(index))),
                        crop_mask_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level]) # crop_mask_img[:,:,[2,1,0]]
                index += 1
            else:
                # print('\tSome No Data pixels in: {:d}, {:d}.  Skipping.'.format(x,y))
                pass

    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()

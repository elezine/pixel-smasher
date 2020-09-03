'''Plot hists of each scene. Uncomment relevatnt areas to switch Between absolute and relative reflectances. Includes switch to compute hist stats for each image. Written for serial loop, not parallel.'''

import os
import os.path as osp
import sys
import getpass
# from multiprocessing import Pool
# import multiprocessing
import numpy as np
import cv2
from matplotlib import pyplot as plt
sys.path.insert(1, '/home/ethan_kyzivat/code/pixel-smasher/codes/scripts/')
from extract_subimgs_single import rescale_reflectance # for relative reflectance
from extract_subimgs_single import rescale_reflectance_equal # for absolute reflectance

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError as e:
    print('Error caught: '+str(e))
    pass

    ## USER PARAMS
btm_percentile=2 # for relative reflectance
top_percentile=95 # for relative reflectance
reflectance_upper=3000 # for absolute reflectance
band_order=(3,2,1)  # 3,2,1 for NRG, 2,1,3 for RGN, 2,1,0 for RGB (original = BGRN) # NOTE this is reversed bc not using cv2 to write out!
ndwi_bands=(1,2) # (1,3) # used to determine maximum or (n-percentile) brightness in scene
save_fig=False # save output hist /stretch fig?
save_hist=True # record histogram for each image, save every save_freq images and at end
hist_length=15000 #15000 #15000 # if using
save_freq=10 # only matters if save_hist is True #30
# n_thread = multiprocessing.cpu_count()
##################

    ## validate I/O
if ~(save_fig or save_hist):
    EnvironmentError('Save_fig or save_hist must be True')

def main():
    """A multi-thread tool to crop sub imags."""
    if getpass.getuser()=='ekyzivat': # on ethan local
        input_folder = 'F:\ComputerVision\Planet'
        save_folder = 'F:\ComputerVision\Planet_sub'
    elif getpass.getuser()=='ethan_kyzivat' or getpass.getuser()=='ekaterina_lezine': # on GCP 
        input_folder = '/data_dir/Scenes'
        save_folder = '/data_dir/other/hists/histsv6_test'
    else: # other
        raise ValueError('input_folder not specified!')
        pass
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
    for root, dirsfoo, file_list in sorted(os.walk(input_folder)): # +'/*SR.tif' # _ instead of dirsfoo
        for x in file_list:  # assume only images in the input_folder # [38:]
            if x.endswith("SR.tif"):
                path = os.path.join(root, x) 
                img_list.append(path)
        break # ignores files in nested dirs
    # img_list = ['/data_dir/Scenes/20190619_191648_25_106f_3B_AnalyticMS_SR.tif'] # for testing
    def update(arg):
        pbar.update(arg)

    # img_list=img_list[:30] # to start in middle
    pbar = ProgressBar(len(img_list))

    # pool = Pool(n_thread)
    hist_results=np.zeros((hist_length,4,len(img_list)), dtype='single') # init, TODO: make dynamic
    j=0
    for path in img_list:
        print(f'Image {j}:\t{path}')
        hist_results[:,:,j] = worker(path, save_folder, crop_sz, step, thres_sz, compression_level) # ,callback=update
        if j % save_freq==0:
            np.save('histograms_temp_serial.npy', hist_results)
            print('Temp histograms saved to hists_temp.npy')
        j+=1

    # pool.close()
    # pool.join()
    print('All subprocesses done.')
    np.save('histograms.npy', hist_results)
    print('Histograms saved to hists.npy')

def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # for relative reflectance
    # reflectance_upper=np.percentile(img[:,:,ndwi_bands], top_percentile) # Compute maximum reflectance from entire scene, not individual subsets
    print(f'\n\nLoaded image:\t{img_name}')
    print(f'Rescaling reflectance to: {reflectance_upper:.1f}\n')
    

    f, ax = plt.subplots(img.shape[2], 2, sharex=True)
    # dfb.rem
    hist=np.zeros((hist_length, img.shape[2])) # TODO: make 1500 dynamic
    for i in range(img.shape[2]):
        if save_fig:
            ax[i,1].hist(img[:,:,i][img[:,:,i]>0].flatten(),bins=np.linspace(0,10000,101))
            ax[i,1].set_title('band: {}'.format(i))
        if save_hist:
            h=[] # init histogram
            h.append(ax[i,1].hist(img[:,:,i][img[:,:,i]>0].flatten(),bins=np.linspace(0,hist_length,hist_length+1))) # HERE figure out how to use plot or create an if branch
            hist[:,i]=h[0][0]
            pass

        # for relative reflectance
    # img=rescale_reflectance(img[:,:,band_order], btm_percentile, top_percentile) 

        # for absolute reflectance

    
    if save_fig:
        f.add_subplot(1,2,1)
        ax[0,0].set_title(img_name)
        img=rescale_reflectance_equal(img[:,:,band_order], reflectance_upper) 
        plt.imshow(img, resample=True)
        plt.savefig(os.path.join(save_folder, img_name.replace('.tif', '_hist.png')))
    plt.close()
    print(f'\t{img_name} hist\tSaved.')

    if save_hist:
        return hist # HERE: which orientation?
    else: 
        return 0

if __name__ == '__main__':
    main()

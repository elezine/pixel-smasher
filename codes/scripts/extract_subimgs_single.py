import os
import os.path as osp
import sys
import getpass
from multiprocessing import Pool
import numpy as np
import cv2
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError as e:
    print('Error caught: '+str(e))
    pass

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

    n_thread = 8 #20
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
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
        #pass # uncomment above two lines for ease of working, if necessary

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)): # +'/*SR.tif'
        for x in file_list:  # assume only images in the input_folder
            if x.endswith("SR.tif"):
                path = os.path.join(root, x) 
                img_list.append(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # for reflectance scaling
    reflectance_lower=0
    reflectance_upper=6000

        ## apply reflectance scaling correction
    image_cal=np.ones(img.shape, dtype='single')
    # ID=filename[:-10]
    # coeffs=hash[ID]
    image_cal=np.minimum((img.astype(np.single)-reflectance_lower)/(reflectance_upper-reflectance_lower), image_cal) ## need to modify if refl_lower is > 0...
    # image=cv2.normalize(image_cal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # this method uses min of empircal data as original min bound, so is not uniform between images
    img=(image_cal*255).astype(np.uint8)

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
            if ~np.any(crop_img==0):
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.tif', '_s{:04d}.png'.format(index))),
                    crop_img[:,:,(2,1,3)], [cv2.IMWRITE_PNG_COMPRESSION, compression_level]) # 2,1,3 for RGN, 2,1,0 for RGB (original = BGRN)
                index += 1
            else:
                print('\tSome No Data pixels in: {:d}, {:d}.  Skipping.'.format(x,y))
                pass

    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()

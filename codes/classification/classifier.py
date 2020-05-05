import os
import os.path as osp
import sys
# import getpass
# from multiprocessing import Pool
import numpy as np
import cv2
from multiprocessing import Pool


# example output paths: /data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984
def classify(pth_in, pth_out, threshold=10):
    '''
    A simple classification function for high-resolution, low-resolution, and  super resolution images.  Takes input path and write To output path (pre-– formatted).

    TODO: mutliple threshold.  Path parsing.  Batch mode.  Post- calibration.
    '''
    img = cv2.imread(pth_in, cv2.IMREAD_UNCHANGED)
    ndvi = np.array(img[:,:,0]-img[:,:,2], dtype='int16')
    bw=ndvi>threshold # output mask from classifier
    # write out
    cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))
if __name__ == '__main__':

        # for testing
    pth_in, pth_out= '/data_dir/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029.png', 'test.png' #'/data_dir/classify/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029C.png'
    print('Running classifier.')
    print('File:\t{}\nOutput:\t{}\n'.format(pth_in, pth_out))
    im_out=classify(pth_in, pth_out)
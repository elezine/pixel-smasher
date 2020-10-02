''' A quick test to classify whatever is in the /data_dir/planet_sub folder. Uncomment classifier.classify to write stretched NDWI instead of mask.'''
import os
import os.path as osp
import sys
# import getpass
# from multiprocessing import Pool
import numpy as np
import cv2
from multiprocessing import Pool
import multiprocessing as mp
import pickle
import pandas as pd
from classifier import classify


# example output paths: /data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: apply calibration if needed, multiple thresh?

# I/O
sourcedir='/data_dir/hold_mod/HR/x4' # /data_dir/hold_mod
outdir='/data_dir/other/classified_test' #'/data_dir/classified/'
apply_radiometric_correction=False # set to zero if already calibrated

# params
thresh=-0.1

# output folder
if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('mkdir [{:s}] ...'.format(outdir))

# auto I/O
if apply_radiometric_correction:
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)

    # for testing #####################
# pth_in, pth_out= '/data_dir/ClassProject/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029.png', 'test.png' #'/data_dir/ClassProject/classify/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029C.png'
# print('Running classifier.')
# print('File:\t{}\nOutput:\t{}\n'.format(pth_in, pth_out))
# im_out=classify(pth_in, pth_out)
    ##################################
    
    # print
print('Starting classification.  Files will be in {}'.format(outdir))
    # loop over files
dirpaths = [f for f in os.listdir(sourcedir) ] # removed: if f.endswith('.png')
num_files = len(dirpaths)
# global results
results = {} # init
# pool = Pool(1) #mp.cpu_count())
for i in range(num_files): # switch for testing # range(30): #
    name = dirpaths[i]
    print(f'Classifying file: {name}')
    # parallel
    # results[i] = pool.apply(classify, args=(os.path.join(sourcedir, name), os.path.join(outdir, name), thresh, None))# , , callback=collect_result
    results[i]=classify(os.path.join(sourcedir, name), os.path.join(outdir, name), thresh, None, method='local')
# pool.close()
# pool.join()
print('All subprocesses done.')
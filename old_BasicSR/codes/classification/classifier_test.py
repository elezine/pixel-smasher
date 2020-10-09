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
from   timeit   import default_timer as timer
from classifier import classify, name_lookup_og_mask, foreground_threshold, buffer_additional, sourcedir_R_mask, sourcedir_R as sourcedir, outdir, apply_radiometric_correction, method, water_index_type


# example output paths: /data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: apply calibration if needed, multiple thresh?

# I/O use to overwrite globals
sourcedir='/data_dir/hold_mod_shield/HR/x4' # /data_dir/hold_mod
outdir='/data_dir/other//data_dir/classified_shield_test_parallel' #'/data_dir/classified/'
sourcedir_R_mask='/data_dir/hold_mod_shield_masks/HR/x4' # need to specify two parent dirs
# apply_radiometric_correction=False # set to zero if already calibrated
# foreground_threshold=127
# buffer_additional=0

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
    
    # timer
    
start = timer()
    # print
print('Starting classification.  Files will be in {}'.format(outdir))
    # loop over files
dirpaths = [f for f in os.listdir(sourcedir) ] # removed: if f.endswith('.png')
num_files = len(dirpaths)
# global results
results = {} # init
pool = Pool(8) # Pool(6) #mp.cpu_count())
for i in range(0, 200): # switch for testing # range(30): num_files
    name = dirpaths[i]
    name_og_mask=name_lookup_og_mask(name)
    print(f'Classifying file: {name}')
        # parallel (uncomment)
    # results[i] = pool.apply(classify, args=(os.path.join(sourcedir, name), os.path.join(outdir, name), thresh, None))# , , callback=collect_result
        # serial (uncomment)
    # results[i]=classify(os.path.join(sourcedir, name), os.path.join(outdir, name), thresh, None, None, True, 'NaN', method, os.path.join(sourcedir_R_mask, name_og_mask), water_index_type)
    results[i] = pool.apply_async(classify, args= (os.path.join(sourcedir, name), os.path.join(outdir, name), thresh, None, None, True, 'NaN', method, os.path.join(sourcedir_R_mask, name_og_mask), water_index_type))
pool.close()
pool.join()
print('All subprocesses done.')

    # combine results
# df = pd.concat(list(results.values()))
# df = pd.concat(list(results.values())[i] for i in range(len(results)))
df = pd.concat(list(results.values())[i].get()[0] for i in range(len(results)))
df
    # timer
end = timer()
print('Elapsed time: {:.2f} minutes.'.format((end - start)/60))
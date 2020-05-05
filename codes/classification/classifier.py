import os
import os.path as osp
import sys
# import getpass
# from multiprocessing import Pool
import numpy as np
import cv2
from multiprocessing import Pool


# example output paths: /data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: apply calibration if needed, multiple thresh?

# I/O
sourcedir_SR='/data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_R='/data_dir/valid_mod' # HERE update
outdir='/data_dir/classify/valid_mod'
up_scale=4
iter=100000 # quick fix to get latest validation image in folder

def group_classify(sourcedir_SR, sourcedir_R, outdir, name, threshold=10): # filstrucutre is pre-defined
    '''
    A simple classification function for high-resolution, low-resolution, and  super resolution images.  Takes input path and write To output path (pre-– formatted).
    '''

        # in paths
    SR_in_pth=sourcedir_SR+os.sep+name+os.sep+name+'_'+str(iter)+'.png'
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), name+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), name+ '.png')
    Bic_in_path=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), name+ '.png')

        # out paths
    SR_out_pth = os.path.join(outdir, 'SR', 'x' + str(up_scale), name+ '.png')
    HR_out_pth = os.path.join(outdir, 'HR', 'x' + str(up_scale), name+ '.png')
    LR_out_pth = os.path.join(outdir, 'LR', 'x' + str(up_scale), name+ '.png')
    Bic_out_path = os.path.join(outdir, 'Bic', 'x' + str(up_scale), name+ '.png')

        # run classification procedure
    classify(SR_in_pth, SR_out_pth)
    

def classify(pth_in, pth_out, threshold=10):
        # classify procedure
    img = cv2.imread(pth_in, cv2.IMREAD_UNCHANGED) # HERE change
    ndvi = np.array(img[:,:,0]-img[:,:,2], dtype='int16')
    bw=ndvi>threshold # output mask from classifier
        # count pixels

        # write out
    cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))
    pass # TODO
if __name__ == '__main__':

        # for testing
    # pth_in, pth_out= '/data_dir/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029.png', 'test.png' #'/data_dir/classify/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029C.png'
    # print('Running classifier.')
    # print('File:\t{}\nOutput:\t{}\n'.format(pth_in, pth_out))
    # im_out=classify(pth_in, pth_out)

        # loop over files
    dirpaths = [f for f in os.listdir(sourcedir_SR) ] # removed: if f.endswith('.png')
    num_files = len(dirpaths)

    for i in [1]: #range(num_files): # switch for testing
        name = dirpaths[i]
        
        print('No.{} -- Classifying {}'.format(i, name)) # printf: end=''
        im_out=group_classify(sourcedir_SR, sourcedir_R, outdir, name)
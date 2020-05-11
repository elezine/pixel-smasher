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


# example output paths: /data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: apply calibration if needed, multiple thresh?

# I/O
sourcedir_SR='/data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx8_PLANET/val_images' #'/data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_R='/data_dir/valid_mod_cal' #'/data_dir/valid_mod' # HERE update
outdir='/data_dir/classify/valid_mod_cal'
up_scale=8
iter=60000 # quick fix to get latest validation image in folder
thresh= [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3] # [-10, -5, -2, 0, 2, 5, 10] #2
apply_radiometric_correction=False # set to zero if already calibrated

# auto I/O
if apply_radiometric_correction:
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)

def group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, threshold=2, hash=None): # filstrucutre is pre-defined
    '''
    A simple classification function for high-resolution, low-resolution, and  super resolution images.  Takes input path and write To output path (pre-– formatted).
    '''
        # init
    int_res=[None, None] + [None]*len(thresh)*4 #intermediate result
        # in paths
    SR_in_pth=sourcedir_SR+os.sep+name+os.sep+name+'_'+str(iter)+'.png'
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), name+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), name+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), name+ '.png')

    # save out put to row
    int_res[0]=i
    int_res[1]=name

    for n in range(len(thresh)):
        current_thresh=thresh[n]
        # print('---------------------------')
            # out paths
        SR_out_pth = os.path.join(outdir, 'SR', 'x' + str(up_scale), name+'_T'+str(current_thresh)+ '.png')
        HR_out_pth = os.path.join(outdir, 'HR', 'x' + str(up_scale), name+'_T'+str(current_thresh)+ '.png')
        LR_out_pth = os.path.join(outdir, 'LR', 'x' + str(up_scale), name+'_T'+str(current_thresh)+ '.png')
        Bic_out_pth = os.path.join(outdir, 'Bic', 'x' + str(up_scale), name+'_T'+str(current_thresh)+ '.png')

            # run classification procedure
        # if 
        if 1==1: #os.path.isfile(Bic_out_pth)==False: # only write if file doesn't exist\ # HERE change back
            if n==0:
                print('No.{} -- Classifying {}: '.format(i, name), end='') # printf: end='' # somehow i is in this functions namespace...?
            int_res[2 + 4*n]=classify(SR_in_pth, SR_out_pth,current_thresh, name, hash)
            int_res[3 + 4*n]=classify(HR_in_pth, HR_out_pth,current_thresh, name, hash)
            int_res[4 + 4*n]=classify(LR_in_pth, LR_out_pth,current_thresh, name, hash)
            int_res[5 + 4*n]=classify(Bic_in_pth, Bic_out_pth,current_thresh, name, hash)
        else:# elif os.path.isfile(saveHRpath+os.sep+filename)==True: 
            if n==0:
                print('Skipping: {}.'.format(name))
        print('{}'.format(current_thresh), end=' ')
    print('')
    return int_res

def classify(pth_in, pth_out, threshold=2, name='NaN', hash=None):
        # classify procedure
    img = cv2.imread(pth_in, cv2.IMREAD_UNCHANGED)

        # rad correction
    if apply_radiometric_correction:
        stretch_multiplier=1
        b=[3,2,4]
        img_uint16=cv2.normalize(img, None, 0, 2**16-1, cv2.NORM_MINMAX, dtype=cv2.CV_16U) #img_uint16=img.astype(np.uint16)
        img_cal=np.array(np.zeros(img.shape), dtype='double')
        ID=name[:-6]
        coeffs=hash[ID]
        for j in range(3):
            img_cal[:,:,j]=img_uint16[:,:,j]*coeffs[b[j]]*255*stretch_multiplier
        img=img_cal.astype(np.uint8)

        #continue
    img=np.int16(img)
    ndwi = (img[:,:,0]-img[:,:,2])/(img[:,:,0]+img[:,:,2])
    bw=ndwi>threshold # output mask from classifier
        # count pixels
    nWaterPix=np.sum(bw)
        # write out
    cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))  # HERE
    #pass # Why is this necessary?  It's not
    return nWaterPix

# def collect_result(result):
#     global results
#     results.append(result)

if __name__ == '__main__':

        # for testing
    # pth_in, pth_out= '/data_dir/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029.png', 'test.png' #'/data_dir/classify/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029C.png'
    # print('Running classifier.')
    # print('File:\t{}\nOutput:\t{}\n'.format(pth_in, pth_out))
    # im_out=classify(pth_in, pth_out)

        # print
    print('Starting classification.  Files will be in {}'.format(outdir))
        # loop over files
    dirpaths = [f for f in os.listdir(sourcedir_SR) ] # removed: if f.endswith('.png')
    num_files = len(dirpaths)
    # global results
    results = {} # init
    pool = Pool(mp.cpu_count())
    for i in range(num_files): #range(num_files): # switch for testing # range(30): #
        name = dirpaths[i]

        # parallel
        results[i] = pool.apply_async(group_classify, args=(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash)).get() # , , callback=collect_result
    pool.close()
    pool.join()
    print('All subprocesses done.')


    # save result
    cols_fmt=['num','name']+['TBD']*len(thresh)*4 # formatted c olumn names
    for n in range(len(thresh)):
        cols_fmt[2 + 4*n]= 'SR'+'_T'+str(thresh[n]) #hr lr bic
        cols_fmt[3 + 4*n]= 'HR'+'_T'+str(thresh[n])
        cols_fmt[4 + 4*n]= 'LR'+'_T'+str(thresh[n])
        cols_fmt[5 + 4*n]= 'Bic'+'_T'+str(thresh[n])
    df = pd.DataFrame(list(results.values()), columns =cols_fmt)
df.to_csv('classification_stats_x'+str(up_scale)+'_'+str(iter)+'.csv') # zip(im_name, hr, lr, bic, sr)

        ## for non- parallel
    #im_out=group_classify(sourcedir_SR, sourcedir_R, outdir, name)
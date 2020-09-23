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


# example output paths: /data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: 9/16: long format? Add kappa, add nearest neighbor upsample?, record overall image ndwi brightness

# I/O
sourcedir_SR='/home/ethan_kyzivat/data_dir/visualization' # note: shuf2k is just a 2000 image shuffling #'/data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_R='/home/ethan_kyzivat/data_dir/valid_mod' #'/data_dir/ClassProject/valid_mod'
outdir='/home/ethan_kyzivat/data_dir/classified/valid_mod'
up_scale=4
for j in ['HR','SR','LR','Bic']:
    os.makedirs(os.path.join(outdir, j, 'x'+str(up_scale)), exist_ok=True)
iter=400000 # quick fix to get latest validation image in folder
thresh= [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3] # [-10, -5, -2, 0, 2, 5, 10] #2
apply_radiometric_correction=False # set to zero if already calibrated

# auto I/O
if apply_radiometric_correction:
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)
else: hash=None

def group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, threshold=2, hash=None): # filstrucutre is pre-defined
    '''
    A simple classification function for high-resolution, low-resolution, and  super resolution images.  Takes input path and write To output path (pre-– formatted).
    '''
        # init
    int_res=[None, None] + [None]*len(thresh)*4 #intermediate result
        # in paths
    SR_in_pth=sourcedir_SR+os.sep+name+'_'+str(iter)+'.png' # HERE changed for seven-steps
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
        if os.path.isfile(Bic_out_pth)==False: # only write if file doesn't exist
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
    if apply_radiometric_correction: #HERE update
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
    img=np.single(img)
    ndwi_bands=(2,1)
    ndwi = (img[:,:,ndwi_bands[1]]-img[:,:,ndwi_bands[0]])/(img[:,:,ndwi_bands[1]]+img[:,:,ndwi_bands[0]]) # NRG images: so 2-0 # RGN images: so (G-N)/(G+N)

        # convert nan to zero
    ndwi[np.isnan(ndwi)]=0 # now, I can ignore RuntimeWarnings about dividing by zero
    
    try:
        bw=ndwi>threshold # output mask from classifier
    except RuntimeWarning:
        pass

        # stats: count pixels, etc
    nWaterPix=np.sum(bw)
    mean_ndwi=np.mean(ndwi)
    median_ndwi=np.median(ndwi)
    min_ndwi=np.min(ndwi)
    max_ndwi=np.max(ndwi)

        # write out ndwi ( for testing)
    # img_ndwi=np.minimum(np.maximum((ndwi+0.4)/0.8, np.zeros(ndwi.shape, dtype=ndwi.dtype)), np.ones(ndwi.shape, dtype=ndwi.dtype))
    # cv2.imwrite(pth_out, img_as_ubyte(img_ndwi))  # HERE np.array(255*bw, 'uint8')

        # write out bw
    cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))  # HERE

    #pass # Why is this necessary?  It's not
    return nWaterPix

# def collect_result(result):
#     global results
#     results.append(result)

if __name__ == '__main__':

        # for testing #####################
    # pth_in, pth_out= '/data_dir/ClassProject/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029.png', 'test.png' #'/data_dir/ClassProject/classify/valid_mod/HR/x4/492899_1166117_2017-05-06_1041_BGRN_Analytic_s0029C.png'
    # print('Running classifier.')
    # print('File:\t{}\nOutput:\t{}\n'.format(pth_in, pth_out))
    # im_out=classify(pth_in, pth_out)
        ##################################
        
        # print
    print('Starting classification.  Files will be in {}'.format(outdir))
    os.makedirs(outdir, exist_ok=True)
        # loop over files
    dirpaths = [f for f in os.listdir(sourcedir_SR) ] # removed: if f.endswith('.png')
    num_files = 50 # len(dirpaths) # HERE change back
    # global results
    results = {} # init
    pool = Pool(mp.cpu_count())
    for i in range(num_files): # switch for testing # range(30):
        name = dirpaths[i].replace('_'+str(iter)+'.png', '') # HERE changed for seven-steps from `dirpaths[i]`

        # parallel
        results[i] = pool.apply_async(group_classify, args=(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash))# , , callback=collect_result
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
    try:
        csv_out='classification_stats_x'+str(up_scale)+'_'+str(iter)+'.csv'
        df.to_csv(csv_out) # zip(im_name, hr, lr, bic, sr)
        print('Saved classification stats csv: {}'.format(csv_out))
    except NameError:
        print('No CSV printed')
        ## for non- parallel
    #im_out=group_classify(sourcedir_SR, sourcedir_R, outdir, name)
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
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


# example output paths: /data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: 9/16: long format? Add kappa, add nearest neighbor upsample?, record overall image ndwi brightness

# I/O
sourcedir_SR='/data_dir/pixel-smasher/experiments/003_ESRGAN_x4_PLANET_pretrainDF2K_wandb_sep6/visualization' # note: shuf2k is just a 2000 image shuffling #'/data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_R='/data_dir/valid_mod' #'/data_dir/ClassProject/valid_mod'
outdir='/data_dir/classified/valid_mod'
up_scale=4
for j in ['HR','SR','LR','Bic']:
    os.makedirs(os.path.join(outdir, j, 'x'+str(up_scale)), exist_ok=True)
iter=400000 # quick fix to get latest validation image in folder
thresh= [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3] # [-10, -5, -2, 0, 2, 5, 10] #2
apply_radiometric_correction=False # set to zero if already calibrated
num_metrics=10  # TAG depends-on-num-metrics

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
    # int_res=[None, None] + [None]*len(thresh)*num_metrics #+ [None]*2 #intermediate result # TAG depends-on-num-metrics
        # in paths
    SR_in_pth=sourcedir_SR+os.sep+name+os.sep+name+'_'+str(iter)+'.png' # HERE changed for seven-steps
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), name+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), name+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), name+ '.png')

    # save out put to row
    # int_res[0]=i
    # int_res[1]=name
        # init empty ClassifierComparison object
    data_frame_out=ClassifierComparison()
    data_frame_out=data_frame_out[0:0]
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
            write=True
            if n==0:
                print('No.{} -- Classifying {}: '.format(i, name), end='') # printf: end='' # somehow i is in this functions namespace...?
        else:# elif os.path.isfile(saveHRpath+os.sep+filename)==True: 
            write=False
            if n==0:
                print('No.{} -- Exists {}: '.format(i, name), end='')
        int_res_SR, bw_SR=classify(SR_in_pth, SR_out_pth,current_thresh, name, write=write, res='SR') # TAG depends-on-num-metrics
        int_res_HR, bw_HR = classify(HR_in_pth, HR_out_pth,current_thresh, name, write=write,res='HR')
        int_res_LR, _ = classify(LR_in_pth, LR_out_pth,current_thresh, name, write=write, res='LR')
        int_res_Bic, bw_Bic = classify(Bic_in_pth, Bic_out_pth,current_thresh, name, write=write,res='Bic')
        int_res_SR.kappa=compute_kappa(bw_HR, bw_SR)
        int_res_Bic.kappa=compute_kappa(bw_HR, bw_Bic)
        # int_res[7 + 10*n]=compute_kappa(bw_HR, bw_Bic)
        data_frame_out=data_frame_out.append(pd.concat([int_res_SR, int_res_HR, int_res_LR, int_res_Bic]))
        print('{}'.format(current_thresh), end=' ')
    print('')
    # concat

    data_frame_out.num=i # broadcast?
    return data_frame_out

def classify(pth_in, pth_out, threshold=2, name='NaN', hash=None, write=True, res='NaN'):
        # classify procedure
    ''' Write= whether or not to write classified file. Returns classified matrix as second output '''
    img = cv2.imread(pth_in, cv2.IMREAD_UNCHANGED)

        # check
    if np.any(img==None): # I have to create my own error bc cv2 wont... :(
        raise ValueError(f'Unable to load image: path doesn\'t exist: {pth_in}')

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

        # stats: count pixels, etc # TAG depends-on-num-metrics
    nWaterPix=np.sum(bw)
    percent_water=nWaterPix/bw.size*100
    mean_ndwi=np.mean(ndwi)
    median_ndwi=np.median(ndwi)
    min_ndwi=np.min(ndwi)
    max_ndwi=np.max(ndwi)

        # write out ndwi ( for testing)
    # img_ndwi=np.minimum(np.maximum((ndwi+0.4)/0.8, np.zeros(ndwi.shape, dtype=ndwi.dtype)), np.ones(ndwi.shape, dtype=ndwi.dtype))
    # cv2.imwrite(pth_out, img_as_ubyte(img_ndwi))  # HERE np.array(255*bw, 'uint8')

        # write out bw
    if write:
        cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))  # HERE

    #define and fill output pandas df - this can all be simplified if I include a keywordarg in the ClassifierComparison class __init__
    dataframe_out=ClassifierComparison()
    dataframe_out.name=name
    dataframe_out.thresh=threshold
    dataframe_out.percent_water=percent_water
    dataframe_out.mean_ndwi=mean_ndwi
    dataframe_out.median_ndwi=median_ndwi
    dataframe_out.min_ndwi=min_ndwi
    dataframe_out.max_ndwi=max_ndwi
    dataframe_out.res=res

    return dataframe_out, bw

# def collect_result(result):
#     global results
#     results.append(result)
def compute_kappa(HR_in, test_in):
    ''' Takes in two matrices, flattens, and computes kappa'''
            # kappa score
    # if img.shape[0]==480: # if SR or HR or Bic image
    kappa=cohen_kappa_score(HR_in.flatten()+1, test_in.flatten()+1) # +1 added to prevent div by zero
    return kappa

class ClassifierComparison(pd.DataFrame):
    def __init__(self, data=[np.nan]*10, index=None, columns=None):
        super().__init__([data], columns=['num', 'name', 'thresh','res','percent_water','mean_ndwi', 'median_ndwi','kappa','min_ndwi','max_ndwi'])
        # TODO further: make is so I can pre-populate columns like thresh etc with the class call. not imp for now
        # Like this: def __init__(self, data=[np.nan]*8, index=None, columns=None, k=np.nan):
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
    num_files = len(dirpaths) # #HERE change back
    # global results
    results = {} # init
    # pool = Pool(mp.cpu_count())
    for i in range(num_files): # switch for testing # range(30):
        name = dirpaths[i].replace('_'+str(iter)+'.png', '') # HERE changed for seven-steps from `dirpaths[i]`

        # parallel
        results[i] = group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash)
        # results[i] = pool.apply_async(group_classify, args=(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash)).get()# , , callback=collect_result
    # pool.close()
    # pool.join()
    print('All subprocesses done.')


    # save result
    # cols_fmt=['num','name']+['TBD']*len(thresh)*num_metrics # formatted c olumn names # TAG depends-on-num-metrics
    # for n in range(len(thresh)):
    #     cols_fmt[2 + 10*n]= 'SR'+'_T'+str(thresh[n]) #hr lr bic
    #     cols_fmt[3 + 10*n]= 'HR'+'_T'+str(thresh[n])
    #     cols_fmt[4 + 10*n]= 'LR'+'_T'+str(thresh[n])
    #     cols_fmt[5 + 10*n]= 'Bic'+'_T'+str(thresh[n])
    #     cols_fmt[6 + 10*n]= 'SR'+'_K'+str(thresh[n]) # kappa
    #     cols_fmt[7 + 10*n]= 'Bic'+'_K'+str(thresh[n]) 
    #     cols_fmt[8 + 10*n]= 'SR'+'_M'+str(thresh[n]) # mean
    #     cols_fmt[9 + 10*n]= 'HR'+'_M'+str(thresh[n])
    #     cols_fmt[10 + 10*n]= 'LR'+'_M'+str(thresh[n])
    #     cols_fmt[11 + 10*n]= 'Bic'+'_M'+str(thresh[n])
    df = pd.concat(list(results.values()))
    try:
        csv_out='classification_stats_x'+str(up_scale)+'_'+str(iter)+'.csv'
        df.to_csv(csv_out) # zip(im_name, hr, lr, bic, sr)
        print('Saved classification stats csv: {}'.format(csv_out))
    except NameError:
        print('No CSV printed')
        ## for non- parallel
    #im_out=group_classify(sourcedir_SR, sourcedir_R, outdir, name)

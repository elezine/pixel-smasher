'''
runs classification script with lots of switchable inputs. If output classified files exists, it loads them instead of computing.
'''

import os
import os.path as osp
import sys
# import getpass
# from multiprocessing import Pool
import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
from multiprocessing import Pool
import multiprocessing as mp
import pickle
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show, ion, ioff
sys.path.insert(1, '/home/ethan_kyzivat/code/pixel-smasher')
from water_mask_funcs_ek import create_buffer_mask


# example output paths: /data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images/716222_1368610_2017-08-27_0e0f_BGRN_Analytic_s0984

# TODO: 9/16:add nearest neighbor upsample?,
# TODO: 10/13: fix sloppy file path parsing at "HERE" tags

# I/O
# sourcedir_SR='/data_dir/pixel-smasher/experiments/003_ESRGAN_x4_PLANET_pretrainDF2K_wandb_sep6/visualization' # note: shuf2k is just a 2000 image shuffling #'/data_dir/ClassProject/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_SR='/mnt/disks/extraspace/pixel-smasher/results/008_ESRGAN_x10_PLANET_noPreTrain_130k_Shorelines_Test/visualization/hold_mod_shield_v2' # from shield2 holdout # '/data_dir/pixel-smasher/results/008_ESRGAN_x10_PLANET_noPreTrain_130k_Test/visualization/hold_mod_shield_v2'
sourcedir_R='/data_dir/hold_mod_scenes-shield-gt-subsets' # should have folders for LR, HR, Bic #'/data_dir/ClassProject/valid_mod' # '/data_dir/hold_mod_shield_v2/'
sourcedir_R_mask='/data_dir/hold_mod_scenes-shield-gt-subsets_masks' # '/data_dir/hold_mod_shield_v2_masks'
outdir='/data_dir/classified_shield_v2/008_ESRGAN_x10_PLANET_noPreTrain_130k_Shorelines_Test/visualization' # for shield # /data_dir/classified_shield/hold_mod # '/data_dir/classified_shield_v2/008_ESRGAN_x10_PLANET_noPreTrain_130k_Test_hold_shield_v2/visualization'
model_suffix='_008_ESRGAN_x10_PLANET_noPreTrain_130k_Shorelines_Test' # suffix applied to end of images when they are run as part of a model run call, rather than a validation routine. manually update when you input input folder. Does no harm if images don't have any suffix. # e.g. '_008_ESRGAN_x10_PLANET_noPreTrain_130k_Test'
up_scale=10
for j in ['HR','SR','LR','Bic']:
    os.makedirs(os.path.join(outdir, j, 'x'+str(up_scale)), exist_ok=True)
iter=400000 # quick fix to get latest validation image in folder
thresh= [0] # [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3] # [-10, -5, -2, 0, 2, 5, 10] #2
apply_radiometric_correction=False # For v1 of applying lookup table values to convert to radiance. Set to zero if already calibrated
num_metrics=10  # TAG depends-on-num-metrics
method='local-masked'
    # I/O for create_buffer_mask function
foreground_threshold=127
buffer_additional=0
ndwi_bands=(2,1) #N,G
water_index_type='ir'
plots_dir=None # '/data_dir/other/classifier_plts/008_ESRGAN_x10_PLANET_noPreTrain_130k_Test_hold_shield_v2_XR' # HERE # set to None to not plot # /data_dir/other/classified_shield_test_plots
n_thread=1 #mp.cpu_count() #mp.cpu_count() # use n_thread > 1 for multiprocessing
save_freq=200 # 150 # HERE

# auto I/O
if apply_radiometric_correction:
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)
else: hash=None
if plots_dir != None:
    os.makedirs(plots_dir, exist_ok=True)
def group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, threshold=0.2, hash=None, method='thresh', sourcedir_R_mask=None): # filstrucutre is pre-defined
    '''
    A simple classification function for high-resolution, low-resolution, and  super resolution images.  Takes input path and write To output path (pre-– formatted).
    '''
        # init
    # int_res=[None, None] + [None]*len(thresh)*num_metrics #+ [None]*2 #intermediate result # TAG depends-on-num-metrics

        # in paths
    # SR_in_pth=sourcedir_SR+os.sep+name+os.sep+name+'_'+str(iter)+'.png' # HERE changed for seven-steps and for Shield holdout
    SR_in_pth=os.path.join(sourcedir_SR, name) # HERE changed for seven-steps and for Shield holdout
    name=name.replace(model_suffix, '').replace('.png', '') # quick fix HERE
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), name+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), name+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), name+ '.png')

                # in paths (masks)
    HR_og_mask_pth_in=os.path.join(sourcedir_R_mask, 'HR', 'x' + str(up_scale), name + '_no_buffer_mask.png') # HERE sloppy quick fix # used to read: name.replace('MS_SR', 'MS_SR_no_buffer_mask')
    LR_og_mask_pth_in=os.path.join(sourcedir_R_mask, 'LR', 'x' + str(up_scale), name + '_no_buffer_mask.png')
    Bic_og_mask_pth_in=os.path.join(sourcedir_R_mask, 'Bic', 'x' + str(up_scale), name + '_no_buffer_mask.png')
    SR_og_mask_pth_in=HR_og_mask_pth_in

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
        int_res_SR, bw_SR=classify(SR_in_pth, SR_out_pth,current_thresh, name, write=write, res='SR', method=method, og_mask_pth_in=SR_og_mask_pth_in, water_index_type=water_index_type) # TAG depends-on-num-metrics
        int_res_HR, bw_HR = classify(HR_in_pth, HR_out_pth,current_thresh, name, write=write,res='HR', method=method, og_mask_pth_in=HR_og_mask_pth_in, water_index_type=water_index_type)
        int_res_LR, bw_LR = classify(LR_in_pth, LR_out_pth,current_thresh, name, write=write, res='LR', method=method, og_mask_pth_in=LR_og_mask_pth_in, water_index_type=water_index_type)
        int_res_Bic, bw_Bic = classify(Bic_in_pth, Bic_out_pth,current_thresh, name, write=write,res='Bic', method=method, og_mask_pth_in= Bic_og_mask_pth_in, water_index_type=water_index_type)
        diff=diff_image(bw_SR, bw_Bic, 0)
        mask=np.isin(diff, (1,3)) # simply uses areas with no SR and Bic overlap
        # mask=(mask_0) | (~mask_0 & ~bw_HR) 
        # mask=(bw_SR | bw_Bic | bw_HR) & ~(bw_SR & bw_Bic & bw_HR) # positive mask to use for kappa prime computation: includes all areas where any 2 or all of SR, Bic, and HR disagree

            # kappa metrics
        int_res_SR.kappa=compute_kappa(bw_HR, bw_SR)
        int_res_Bic.kappa=compute_kappa(bw_HR, bw_Bic)
        int_res_SR.kappa_p=compute_kappa_p(bw_HR, bw_SR, mask)
        int_res_Bic.kappa_p=compute_kappa_p(bw_HR, bw_Bic, mask)

            # Overall accuracy metrics
        int_res_SR.accuracy=compute_accuracy(bw_HR, bw_SR)
        int_res_Bic.accuracy=compute_accuracy(bw_HR, bw_Bic)
        int_res_SR.accuracy_p=compute_accuracy_p(bw_HR, bw_SR, mask)
        int_res_Bic.accuracy_p=compute_accuracy_p(bw_HR, bw_Bic, mask)

        # int_res[7 + 10*n]=compute_kappa(bw_HR, bw_Bic)
        data_frame_out=data_frame_out.append(pd.concat([int_res_SR, int_res_HR, int_res_LR, int_res_Bic]))
        print('{}'.format(current_thresh), end=' ')
    print('')
    # concat

    data_frame_out.num=i # broadcast?
    return data_frame_out

def classify(pth_in, pth_out, threshold=2, name='NaN', hash=None, write=True, res='NaN', method='thresh', og_mask_pth_in=None, water_index_type='ndwi'):
        # classify procedure
    ''' 
    Write:              whether or not to write classified file. Returns classified matrix  as second output
    Method:             {thresh, local, local-masked}, where thresh is a series of thresholds, and local uses otsu or similar with adaptive binarization. OG mask only used for local-masked method
    og_mask_pth_in:     path to a priori mask (Pekel)
    water_index_type:   {'ndwi', 'ir'} > from env variable
    '''
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

        # extract water index
    if water_index_type=='ndwi':
        water_index = (img[:,:,ndwi_bands[1]]-img[:,:,ndwi_bands[0]])/(img[:,:,ndwi_bands[1]]+img[:,:,ndwi_bands[0]]) # NRG images: so 2-0 # RGN images: so (G-N)/(G+N)
            # convert nan to zero
        water_index[np.isnan(water_index)]=0 # now, I can ignore RuntimeWarnings about dividing by zero
             # operator overloading to make < = > for IR, not NDWI # https://www.geeksforgeeks.org/operator-overloading-in-python/
        class compare:
            def __init__(self, a):
                self.a = a

            # reverse greater than symbol!
            def __gt__(self, o):
                return self.a > o.a 
    elif water_index_type=='ir':
            # convert nan to zero
        if method=='local': 
            raise RuntimeError('Threshold needs to be updated for IR...')
        water_index = img[:,:,ndwi_bands[0]]
        water_index[np.isnan(water_index)]=255 # now, I can ignore RuntimeWarnings about dividing by zero

        class compare:
            def __init__(self, a):
                self.a = a

            # reverse greater than symbol!
            def __gt__(self, o):
                return self.a <= o.a 
            #
    else: raise ValueError('Index not recognized (EK).')

    if os.path.exists(pth_out)==False: # if output image doesn't already exist
            # binarize image    
        if method=='thresh': 
            try:
                bw=compare(water_index)>compare(threshold) # output mask from classifier
            except RuntimeWarning:
                pass

        elif method=='local': 
            # bw=cv2.adaptiveThreshold(ndwi,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
            thresh=threshold_local(water_index, 75, offset=0, method='gaussian')
            bw=compare(water_index)>compare(thresh)
        elif method =='local-masked':
            '''
            This method takes in a priori water mask (Pekel mask) and runs an Otsu-like threshold for each buffered mask region in the image. Thus, it will almost always detect water if the a priori mask has water, but will never detect water in non-water regions of the a-priori mask. In other words, it is more likely to have false negatives, esp. for small water bodies.
            '''
            # foreground_threshold=127
            # buffer_additional=0 # Note: output mask will be boolean dtype
            if og_mask_pth_in==None or og_mask_pth_in==np.nan:
                raise ValueError('EK: You didn\'t specify a valid og_mask path!')
            og_mask = cv2.imread(og_mask_pth_in, cv2.IMREAD_UNCHANGED)
            if np.any(og_mask==None): # I have to create my own error bc cv2 wont... :(
                raise ValueError(f'Unable to load image: path doesn\'t exist: {pth_in}')

            buffer_mask=create_buffer_mask(og_mask, foreground_threshold, buffer_additional)

                # classifier
            labeled = measure.label(buffer_mask, background=0, connectivity=2)
            bw = np.full(labeled.shape, False)
            if np.all(buffer_mask==0):
                pass
            elif water_index.min()-water_index.max() == 0:
                print('Uniform image. Setting = to all water.')
                bw = np.full(labeled.shape, True)
                pass
            else: # continue to classifier
                copy = np.full(labeled.shape, False)
                regions = measure.regionprops(labeled)
                for x,region in enumerate(regions):
                    # coords = region.coords
                    # i = coords[:,0]
                    # j = coords[:,1]

                    # copy[i,j] = True

                    dist=0 # being lazy and modified from create_buffer_mask_fxn
                    bbox_coords = region.bbox #(min_row, min_col, max_row, max_col)
                    if bbox_coords[0] - dist >= 0:
                        bbox_i_min = bbox_coords[0] - dist
                    else: 
                        bbox_i_min = 0
                    if bbox_coords[1] - dist >= 0:
                        bbox_j_min = bbox_coords[1] - dist
                    else:
                        bbox_j_min = 0
                    if bbox_coords[2] + dist <= copy.shape[0]:
                        bbox_i_max = bbox_coords[2] + dist
                    else:
                        bbox_i_max = copy.shape[0]
                    if bbox_coords[3] + dist <= copy.shape[1]:
                        bbox_j_max = bbox_coords[3] + dist
                    else:
                        bbox_j_max = copy.shape[1]

                    copy_x = copy[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max]
                    ndwi_x = water_index[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max]
                    thresh_x=threshold_otsu(water_index)
                    copy_x=compare(ndwi_x)>compare(thresh_x)
                    copy[bbox_i_min:bbox_i_max, bbox_j_min:bbox_j_max] = copy_x
                    #bounds = find_boundaries(pekel_copy)
                    #pekel_copy[bounds] = 1

                    bw = bw | copy

            # for ...
        else: print(f'Unknown classifier method: {method}')
    else: # if image already exists
        bw=cv2.imread(pth_out, cv2.IMREAD_UNCHANGED)
        bw=bw > foreground_threshold

        # plotting (uncomment for real HERE)
    if (plots_dir != None): # (res==SR) 
        fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        axs[0].imshow(img/255), axs[0].set_title(res+' Image')
        axs[1].imshow(water_index, cmap='bone'), axs[1].set_title(res+' Water index')
        axs[3].imshow(bw, cmap='Greys_r'), axs[3].set_title(res+ ' BW')
        axs[2].imshow(og_mask, cmap='Greys_r'), axs[2].set_title('A priori BW')
        # show()
        plot_pth=os.path.join(plots_dir, 'PLOT_' + os.path.basename(pth_out).replace('.png', '_'+res+'.png'))
        fig.savefig(plot_pth)
        plt.close()

            ## plotting scrap
        #     if (res==ires) & (plots_dir != None):
        #     for k, ires in enumerate(['LR','Bic','HR','SR']):
        #     fig, axs = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
        #     axs[0+k].imshow(img/255), axs[0].set_title(ires+'Image')
        #     axs[1+k].imshow(water_index, cmap='bone'), axs[1].set_title('Water index')
        #     axs[3+k].imshow(bw, cmap='Greys_r'), axs[3].set_title(ires + 'BW')
        #     axs[2+k].imshow(og_mask, cmap='Greys_r'), axs[2].set_title('A priori BW')
        #     # show()
        #     # plot_pth=os.path.join(plots_dir, 'PLOT_' + os.path.basename(pth_out).replace('.png', '_'+ires+'.png'))
        # plot_pth=os.path.join(plots_dir, 'PLOT_' + os.path.basename(pth_out))
        # fig.savefig(plot_pth)
        # plt.close()
        # stats: count pixels, etc # TAG depends-on-num-metrics
    nWaterPix=np.sum(bw)
    percent_water=nWaterPix/bw.size*100
    mean_ndwi=np.mean(water_index)
    median_ndwi=np.median(water_index)
    min_ndwi=np.min(water_index)
    max_ndwi=np.max(water_index) # HERE: remove when done testing...

        # write out ndwi ( for testing)         
    # img_ndwi=np.minimum(np.maximum((ndwi+0.4)/0.8, np.zeros(ndwi.shape, dtype=ndwi.dtype)), np.ones(ndwi.shape, dtype=ndwi.dtype))
    # cv2.imwrite(pth_out, img_as_ubyte(img_ndwi))  # HERE np.array(255*bw, 'uint8')

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
    
        # write out bw
    if (write) and (os.path.exists(pth_out)==False):
        cv2.imwrite(pth_out, np.array(255*bw, 'uint8'))  # HERE

    return dataframe_out, bw

# def collect_result(result):
#     global results
#     results.append(result)

def compute_accuracy(HR_in, test_in):
    ''' Takes in two matrices, flattens, and computes overall accuracy (percent agreement)'''
    OA=accuracy_score(HR_in.flatten(), test_in.flatten()) 
    return OA

def compute_accuracy_p(HR_in, test_in, mask):
    ''' Takes in two matrices, flattens, and computes overall accuracy prime, a measure of overall accuracy but only applied to a masked portion of the data'''
    OA_p=accuracy_score(HR_in[mask].flatten(), test_in[mask].flatten()) 
    return OA_p ## HERE debug

def compute_kappa(HR_in, test_in):
    ''' Takes in two matrices, flattens, and computes kappa'''
            # kappa score
    # if img.shape[0]==480: # if SR or HR or Bic image
    kappa=cohen_kappa_score(HR_in.flatten(), test_in.flatten()) # +1 added to prevent div by zero
    return kappa

def compute_kappa_p(HR_in, test_in, mask):
    ''' Takes in two matrices, a positive (keep) pixel mask, then flattens, and computes kappa prime, a measure of kappa's coefficient but only applied to a masked portion of the data'''
            # kappa score
    # if img.shape[0]==480: # if SR or HR or Bic image
    kappa_p=cohen_kappa_score(HR_in[mask].flatten(), test_in[mask].flatten()) 
    return kappa_p ## HERE debug

def diff_image(SR,Bic, foreground_threshold):
    '''
    Computes a difference image-like representation between two maps.
    Output: 0 = both agree negative, 1 = Bic is positive, 2 = both agree positive, 3 = SR is positive
    foreground_threshold is used as value to split boolean on. Set to zero if working with Boolean arrays.
    '''
    diff=np.full(SR.shape, 0, dtype='uint8')
    diff[(SR>foreground_threshold) & (Bic>foreground_threshold)]=2 # SR and Bic == water
    diff[(SR>foreground_threshold) & (Bic<=foreground_threshold)]=3 # SR == water and Bic == land
    diff[(SR<=foreground_threshold) & (Bic>foreground_threshold)]=1 # SR == land and Bic == water
    return diff
class ClassifierComparison(pd.DataFrame):
    def __init__(self, data=[np.nan]*13, index=None, columns=None):
        super().__init__([data], columns=['num', 'name', 'thresh','res','percent_water','mean_ndwi', 'median_ndwi','accuracy','accuracy_p','kappa','kappa_p','min_ndwi','max_ndwi'])
        # TODO further: make is so I can pre-populate columns like thresh etc with the class call. not imp for now
        # Like this: def __init__(self, data=[np.nan]*8, index=None, columns=None, k=np.nan):

    # scratch paper for class inheritance...
# class ClassifierComparison(pd.DataFrame):
#     def __init__(self, foo=None,glue=None, index=None, columns=None):
#         super().__init__([foo, glue], columns=['foo','glue'],index=None)
#         # TODO further: make is so I can pre-populate columns like thresh etc with the class call. not imp for now
#         # Like this: def __init__(self, data=[np.nan]*8, index=None, columns=None, k=np.nan):

# class ClassifierComparison(pd.DataFrame):
#     def __init__(self, foo=None, glue=None, data=[np.nan]*2,index=None, columns=None):
#         super().__init__(columns=['foo','glue'],index=None)
#         # self.foo=foo
#         # self.glue=glue
#         self['foo']=foo
#         self['glue']=glue 

def name_lookup_og_mask(name_scene):
    '''
    Simple string replacement to go from i.e. 20200829_200110_99_105e_3B_AnalyticMS_SR_s0656 >>> 20200829_200110_99_105e_3B_AnalyticMS_SR_no_buffer_mask_s0656
    '''
    name_mask_scene = name_scene.replace('_SR_s', '_SR_no_buffer_mask_s')
    return name_mask_scene

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
    if n_thread>1: 
        pool = Pool(n_thread)
    for i in range(0, num_files): #range(num_files): # switch for testing # range(30): # HERE switch
        name = dirpaths[i].replace('_'+str(iter)+'.png', '') # HERE changed for seven-steps from `dirpaths[i]`
        name_og_mask=name_lookup_og_mask(name)
        ############## testing
        # if '20170708_181118_102a_3B_AnalyticMS_SR_s0244' not in name:
        #     continue
        ######################
        if n_thread==1:    # serial
            results[i] = group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash, method, sourcedir_R_mask)
            if i % save_freq == 0:
                df = pd.concat(list(results.values())[i] for i in range(len(results)))
            
        else: # parallel
            results[i] = pool.apply_async(group_classify, args=(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash, method, sourcedir_R_mask))# , , callback=collect_result # no .get()
            if i % save_freq == 0:
                df = pd.concat(list(results.values())[i].get() for i in range(len(results)))
        if i % save_freq == 0: # common to parallel and serial branches
            csv_out='classification_stats_x'+str(up_scale)+'_'+method+'_'+str(iter)+'_'+water_index_type+'_tmp.csv'
            df.to_csv(csv_out) # zip(im_name, hr, lr, bic, sr)
            print('Saved temp. classification stats (length {}) csv: {}'.format(df.shape[0], csv_out))
            del df
    print('All subprocesses done.')

        # concat results
    if n_thread>1: 
        pool.close()
        pool.join()
        df = pd.concat(list(results.values())[i].get() for i in range(len(results)))
    else:
        df = pd.concat(list(results.values())[i] for i in range(len(results)))

        # save results
    try:
        csv_out='classification_stats_x'+str(up_scale)+'_'+method+'_'+str(iter)+'_'+water_index_type+'.csv'
        df.to_csv(csv_out) # zip(im_name, hr, lr, bic, sr)
        print('Saved classification stats csv: {}'.format(csv_out))
    except NameError:
        print('No CSV printed')
        ## for non- parallel
    #im_out=group_classify(sourcedir_SR, sourcedir_R, outdir, name)

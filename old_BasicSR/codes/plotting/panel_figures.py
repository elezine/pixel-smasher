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
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show, ion, ioff
sys.path.insert(1, '/home/ethan_kyzivat/code/pixel-smasher')
sys.path.insert(1, '/home/ethan_kyzivat/code/pixel-smasher/old_BasicSR/codes/classification')
from classifier import sourcedir_SR, sourcedir_R, sourcedir_R_mask, outdir, up_scale, foreground_threshold, ndwi_bands, water_index_type, name_lookup_og_mask, diff_image, model_suffix
import matplotlib.colors as colors

'''
Script based off of classifier to make image panel figures. Takes input paths from classifier.py
'''

# I/O
for j in ['HR','SR','LR','Bic']:
    os.makedirs(os.path.join(outdir, j, 'x'+str(up_scale)), exist_ok=True)
iter=400000 # quick fix to get latest validation image in folder
thresh= [0] # [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3] # [-10, -5, -2, 0, 2, 5, 10] #2
apply_radiometric_correction=False # For v1 of applying lookup table values to convert to radiance. Set to zero if already calibrated
plots_dir='/data_dir/other/classifier_plts/008/008_ESRGAN_x10_PLANET_noPreTrain_130k_Shorelines_Test_panel_figs_zoom35' # HERE # set to None to not plot # /data_dir/other/classified_shield_test_plots # 008_ESRGAN_x10_PLANET_noPreTrain_130k_Test_hold_shield_v2_XR_panel_figs_v2_highres
method='local-masked'
zoom=True # whether or not to zoom in before making panel figures
zoom_percent=35
n_thread=mp.cpu_count() #mp.cpu_count() # use n_thread > 1 for multiprocessing
    # I/O for create_buffer_mask function
# buffer_additional=0

# auto I/O
plots_dir=os.path.join(plots_dir, 'x'+str(up_scale))
if apply_radiometric_correction:
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)
else: hash=None
os.makedirs(plots_dir, exist_ok=True)
def group_plot(i, sourcedir_SR, sourcedir_R, outdir, name, threshold=0.2, hash=None, method='thresh', sourcedir_R_mask=None): # filstrucutre is pre-defined
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
        # int_res_SR, bw_SR=classify(SR_in_pth, SR_out_pth,current_thresh, name, write=write, res='SR', method=method, og_mask_pth_in=SR_og_mask_pth_in, water_index_type=water_index_type) # TAG depends-on-num-metrics
        # int_res_HR, bw_HR = classify(HR_in_pth, HR_out_pth,current_thresh, name, write=write,res='HR', method=method, og_mask_pth_in=HR_og_mask_pth_in, water_index_type=water_index_type)
        # int_res_LR, bw_LR = classify(LR_in_pth, LR_out_pth,current_thresh, name, write=write, res='LR', method=method, og_mask_pth_in=LR_og_mask_pth_in, water_index_type=water_index_type)
        # int_res_Bic, bw_Bic = classify(Bic_in_pth, Bic_out_pth,current_thresh, name, write=write,res='Bic', method=method, og_mask_pth_in= Bic_og_mask_pth_in, water_index_type=water_index_type)
        # int_res_SR.kappa=compute_kappa(bw_HR, bw_SR)
        # int_res_Bic.kappa=compute_kappa(bw_HR, bw_Bic)

        ## new
        # tmp_output=map(cv2.imread, ((pth_in, cv2.IMREAD_UNCHANGED) for pth_in in [SR_out_pth, HR_out_pth, LR_out_pth, Bic_out_pth]))
        
        tmp_output_XR_in=[]
        for pth_in in [HR_in_pth, SR_in_pth, Bic_in_pth, LR_in_pth]:
            tmp_output_XR_in.append(cv2.imread(pth_in, cv2.IMREAD_UNCHANGED))
            if np.any(tmp_output_XR_in[-1]==None): # I have to create my own error bc cv2 wont... :(
                raise ValueError(f'Unable to load image: path doesn\'t exist: {pth_in}')
        # bw_SR, bw_HR, bw_LR, bw_Bic=tmp_output
        
        tmp_output_XR_mask=[]
        for pth_in in [HR_out_pth, SR_out_pth, Bic_out_pth, LR_out_pth]:
            tmp_output_XR_mask.append(cv2.imread(pth_in, cv2.IMREAD_UNCHANGED))
            if np.any(tmp_output_XR_mask[-1]==None): # I have to create my own error bc cv2 wont... :(
                raise ValueError(f'Unable to load image: path doesn\'t exist: {pth_in}')
        # bw_SR, bw_HR, bw_LR, bw_Bic=tmp_output
        
        tmp_output_og=[]
        for pth_in in [HR_og_mask_pth_in, SR_og_mask_pth_in, Bic_og_mask_pth_in, LR_og_mask_pth_in]:
            tmp_output_og.append(cv2.imread(pth_in, cv2.IMREAD_UNCHANGED))
            if np.any(tmp_output_og[-1]==None): # I have to create my own error bc cv2 wont... :(
                raise ValueError(f'Unable to load image: path doesn\'t exist: {pth_in}')
        # og_SR, og_HR, og_LR, og_Bic=tmp_output

        ## make plots in grid

        if (plots_dir != None): # (res==SR) 
            fs=14 # font size
            zb=None # Zoom bounds (100.5, 150.5, 150.5,100.5) # extent # None # L,R,B,T # Note: use plt.xlim to control extent
            if 1==1:
                plot_order=[0,3,2,1] # HR, SR, Cub, LR on btm says Katia; Larry says HR, LR, CR, SR
                fig, axs = plt.subplots(4, 3, figsize=(6, 8), constrained_layout=True) # sharex=True
                cmap_mask = colors.ListedColormap(['black', '#2390D2'])
                vmin=150
                vmax=255
                for k, ires in enumerate(['HR', 'SR','CR','LR']): # default order in list
                    dims=tmp_output_XR_in[k].shape[0:2]
                    if zoom==True:
                        zb=[int(dims[1]*zoom_percent/100), int(dims[1]*(100-zoom_percent)/100), int(dims[0]*(100-zoom_percent)/100), int(dims[0]*zoom_percent/100)]
                    else:
                        zb=[0, int(dims[1]), int(dims[0]), 0] # no crop
                    axs[plot_order[k],0].imshow(tmp_output_XR_in[k][zb[3]:zb[2],zb[0]:zb[1],[2,1,0]]/255, extent=zb, resample=False, vmin=vmin, vmax=vmax), axs[0, 0].set_title('Image', fontsize=fs+4), axs[plot_order[k],0].set_axis_off() #axs[k,0].axes.xaxis.set_visible(False), axs[k,0].axes.yaxis.set_visible(False), # reverse order to make image red, not blue, (RGB=NGR) bc cv2...
                    axs[plot_order[k],1].imshow(tmp_output_XR_in[k][zb[3]:zb[2],zb[0]:zb[1],ndwi_bands[0]], extent=zb, resample=False, cmap='bone', vmin=vmin, vmax=vmax), axs[0,1].set_title('NIR band', fontsize=fs+4), axs[plot_order[k],1].set_axis_off() #axs[k,1].axes.yaxis.set_visible(False), axs[k,1].axes.xaxis.set_visible(False) # try ax.set_axis_off()
                    # axs[k,2].imshow(tmp_output_og[k], cmap='Greys_r'), axs[0,2].set_title('A priori BW'), axs[k,0].axes.yaxis.set_visible(False)
                    axs[plot_order[k],2].imshow(tmp_output_XR_mask[k][zb[3]:zb[2],zb[0]:zb[1]]>foreground_threshold, extent=zb, resample=False, cmap=cmap_mask), axs[0,2].set_title('Water mask', fontsize=fs+4), axs[plot_order[k],2].set_axis_off() #axs[k,2].axes.yaxis.set_visible(False), axs[k,2].axes.xaxis.set_visible(False) # 'Greys_r'
                    
                        # add text
                    # axs[k,0].set_ylabel(ires, fontsize=fs+2)
                    axs[plot_order[k],0].text(-0.05, .5, ires, transform=axs[plot_order[k],0].transAxes, ha='right', va='center', size=fs+4, rotation='vertical')
                # show()
                plot_pth=os.path.join(plots_dir, 'PLOT_' + name + '.png')
                fig.savefig(plot_pth, dpi=300); fig.savefig(plot_pth.replace('.png','.pdf'), dpi=300)
                plt.close()

            ## make second diff plot
            if 1==1:
                    # zoom math
                dims=tmp_output_XR_in[0].shape[0:2]
                if zoom==True:
                    zb=[int(dims[1]*zoom_percent/100), int(dims[1]*(100-zoom_percent)/100), int(dims[0]*(100-zoom_percent)/100), int(dims[0]*zoom_percent/100)]
                else:
                    zb=[0, int(dims[1]), int(dims[0]), 0] # no crop
                xt=[zb[0],zb[1],zb[3],zb[2]]
                
                    # make and plot diff image
                diff=np.full(tmp_output_XR_mask[0].shape, 0, dtype='uint8')
                diff=diff_image(tmp_output_XR_mask[1], tmp_output_XR_mask[2], foreground_threshold)
                # cmap1 = colors.ListedColormap(['black', '#E9C46A', 'white', '#457b9d'])
                cmap1 = colors.ListedColormap(['black', '#B07F3E', '#2390D2', '#6BCAD0'])
                plt.imshow(diff[zb[3]:zb[2],zb[0]:zb[1]], extent=xt, cmap=cmap1,vmin=0, vmax=3, origin='lower') # vmin=0, vmax=3, 
                plt.xlim(xt[:2])
                plt.ylim(xt[2:])
                    # add contour lines (perimeters) for HR boundary
                perims=np.concatenate(measure.find_contours(tmp_output_XR_mask[0], foreground_threshold))
                plt.plot(perims[::1,1], perims[::1,0], '.r', markersize=2.0)
                    # save plot
                plot_pth=os.path.join(plots_dir, 'DIFF_' + name + '.png')
                plt.gca().invert_yaxis()
                plt.gca().set_axis_off()
                plt.gcf().tight_layout()
                plt.savefig(plot_pth, dpi=300); plt.savefig(plot_pth.replace('.png','.pdf'), dpi=300)
                plt.close()

        # int_res[7 + 10*n]=compute_kappa(bw_HR, bw_Bic)
        # data_frame_out=data_frame_out.append(pd.concat([int_res_SR, int_res_HR, int_res_LR, int_res_Bic]))
        print('{}'.format(current_thresh), end=' ')
    print('')
    # concat

    data_frame_out.num=i # broadcast?
    return data_frame_out

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
    print('Starting panel figs.  Files will be in {}'.format(plots_dir))
    os.makedirs(outdir, exist_ok=True)
        # loop over files
    dirpaths = [f for f in os.listdir(sourcedir_SR) if f.endswith('.png')]
    num_files = len(dirpaths) # #HERE change back
    # global results
    results = {} # init
    if n_thread>1: 
        pool = Pool(n_thread) # Pool(2) # Pool(mp.cpu_count())
    for i in range(0, num_files): #range(num_files): # switch for testing # range(30): # HERE switch
        name = dirpaths[i].replace('_'+str(iter)+'.png', '') # HERE changed for seven-steps from `dirpaths[i]`
        name_og_mask=name_lookup_og_mask(name)
                
        ############## testing
        # if '20170708_181118_102a_3B_AnalyticMS_SR_s0244' not in name:
        #     continue
        ######################

            # serial
        # results[i] = group_classify(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash, method, sourcedir_R_mask)

            # parallel
        if n_thread>1: 
            pool.apply_async(group_plot, args=(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash, method, sourcedir_R_mask))# , , callback=collect_result # no .get()
        else:
            group_plot(i, sourcedir_SR, sourcedir_R, outdir, name, thresh, hash, method, sourcedir_R_mask)# , , callback=collect_result # no .get()
    if n_thread>1: 
        pool.close()
        pool.join()
    print('All subprocesses done.')

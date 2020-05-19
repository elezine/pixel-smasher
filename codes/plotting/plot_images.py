#%% imports
#%matplotlib inline
'''function for generating sequence of plots (LR, HR, Bic, SR, classified_threshold_n) for a given image names'''

import os
import sys
import cv2
import numpy as np
import getpass
import pickle
from scipy import stats as st
import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%% I/O
sourcedir_SR='/data_dir/pixel-smasher/experiments/003_RRDB_ESRGANx4_PLANET/val_images'
sourcedir_R='/data_dir/valid_mod'
classdir='/data_dir/classify/valid_mod'
up_scale=4
current_thresh=0.2
iter=60000 # quick fix to get latest validation image in folder
rescale_factor=8 # quick and dirty rescale
names=['622194_1368808_2017-07-14_102d_BGRN_Analytic_s0562', '622194_1368808_2017-07-14_102d_BGRN_Analytic_s0699', '584870_1368808_2017-06-27_1038_BGRN_Analytic_s0110']
#    

# clouds: 660535_1066622_2017-08-01_1041_BGRN_Analytic_s1002 660535_1066622_2017-08-01_1041_BGRN_Analytic_s1002 581200_1368610_2017-06-26_1030_BGRN_Analytic_s0389
#%% loop
fig3, axes = plt.subplots(nrows=len(names),ncols=5, figsize=plt.figaspect(len(names)/5*1))

for i in range(len(names)):
        
    # paths
    SR_in_pth=os.path.join(sourcedir_SR,names[i],names[i]+'_'+str(iter)+'.png')
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), names[i]+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), names[i]+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), names[i]+ '.png')
    class_SR_in_pth=os.path.join(classdir, 'SR', 'x' + str(up_scale), names[i]+'_T'+str(current_thresh)+ '.png')

    # load
    SR_in = cv2.imread(SR_in_pth)
    HR_in = cv2.imread(HR_in_pth)
    LR_in = cv2.imread(LR_in_pth)
    Bic_in = cv2.imread(Bic_in_pth)
    class_SR_in = cv2.imread(class_SR_in_pth, cv2.IMREAD_UNCHANGED) # load as greyscale
    # class_HR_in = cv2.imread(SR_in_pth, cv2.IMREAD_UNCHANGED)

# TODO: plot accuracy metrics, too
    axes[i,0].imshow(SR_in*rescale_factor) # norme=
    axes[i,1].imshow(HR_in*rescale_factor)
    axes[i,2].imshow(LR_in*rescale_factor)
    axes[i,3].imshow(Bic_in*rescale_factor)
    axes[i,4].imshow(class_SR_in, cmap='Greys')
    
    for j in range(5):
        # plot
        # fig=plt.figure()
        # fig.add_subplot(251)
        # axes[i,j].imshow(cv2.normalize(SR_in, norm_type=cv2.NORM_MINMAX))
        # axes[i,j].imshow(cv2.normalize(SR_in, None, 1.0, 0.0, cv2.NORM_L1))
        # axes.tick_params...
        # fig3.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
        axes[i,j].axis('off')
        plt.axis('off')

#%% plot and save
fig3.tight_layout()
plt.savefig('fig3')
print('done')
    

# %%

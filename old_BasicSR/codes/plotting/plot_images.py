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
labels=['A','B','C']
ylabels=('HR','LR','Bic','SR','SR water mask')
za=(125, 380, 200) #100 # zoom bound A
zb=(225, 480, 300) #300; # zoom bound B
#    

# clouds: 660535_1066622_2017-08-01_1041_BGRN_Analytic_s1002 660535_1066622_2017-08-01_1041_BGRN_Analytic_s1002 581200_1368610_2017-06-26_1030_BGRN_Analytic_s0389
#%% loop
fig1, axes = plt.subplots(ncols=len(names),nrows=5, figsize=plt.figaspect(5/len(names)*1))

for i in range(len(names)):
        
    # paths
    SR_in_pth=os.path.join(sourcedir_SR,names[i],names[i]+'_'+str(iter)+'.png')
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), names[i]+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), names[i]+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), names[i]+ '.png')
    class_SR_in_pth=os.path.join(classdir, 'SR', 'x' + str(up_scale), names[i]+'_T'+str(current_thresh)+ '.png')

    # load
    SR_in = cv2.imread(SR_in_pth)[:,:,(0,2,1)]
    HR_in = cv2.imread(HR_in_pth)[:,:,(0,2,1)]
    LR_in = cv2.imread(LR_in_pth)[:,:,(0,2,1)]
    Bic_in = cv2.imread(Bic_in_pth)[:,:,(0,2,1)]
    class_SR_in = cv2.imread(class_SR_in_pth, cv2.IMREAD_UNCHANGED) # load as greyscale
    # class_HR_in = cv2.imread(SR_in_pth, cv2.IMREAD_UNCHANGED)

# TODO: plot accuracy metrics, too
    axes[0,i].imshow(HR_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor)
    axes[1,i].imshow(LR_in[np.int(za[i]/up_scale):np.int(zb[i]/up_scale),np.int(za[i]/up_scale):np.int(zb[i]/up_scale):]*rescale_factor)
    axes[2,i].imshow(Bic_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor) # <----- HERE: fix scaling for LR !
    axes[3,i].imshow(SR_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor, extent=(100.5, 150.5, 150.5,100.5)) # norm=
    axes[4,i].imshow(class_SR_in[za[i]:zb[i],za[i]:zb[i]:], cmap='Greys')
    
    

    for j in range(5):
        # plot
        # fig=plt.figure()
        # fig.add_subplot(251)
        # axes[i,j].imshow(cv2.normalize(SR_in, norm_type=cv2.NORM_MINMAX))
        # axes[i,j].imshow(cv2.normalize(SR_in, None, 1.0, 0.0, cv2.NORM_L1))
        # axes.tick_params...
        # fig1.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
        
        # axes[j,i].axis('off')
        axes[j,i].xaxis.set(ticks=(), ticklabels=())
        axes[j,i].yaxis.set(ticks=(), ticklabels=())
        # plt.axis('off')
        if i==0:
            axes[j,i].set(ylabel=ylabels[j])
        if j==0:
            axes[j,i].set(title=labels[i]) # (title=names[i]
#%% second loop
za=(0,0,0) #100 # zoom bound A
zb=(480, 480, 480) #300; # zoom bound 

fig2, axes2 = plt.subplots(ncols=len(names),nrows=5, figsize=plt.figaspect(5/len(names)*1))

for i in range(len(names)):
        
    # paths
    SR_in_pth=os.path.join(sourcedir_SR,names[i],names[i]+'_'+str(iter)+'.png')
    HR_in_pth=os.path.join(sourcedir_R, 'HR', 'x' + str(up_scale), names[i]+ '.png')
    LR_in_pth=os.path.join(sourcedir_R, 'LR', 'x' + str(up_scale), names[i]+ '.png')
    Bic_in_pth=os.path.join(sourcedir_R, 'Bic', 'x' + str(up_scale), names[i]+ '.png')
    class_SR_in_pth=os.path.join(classdir, 'SR', 'x' + str(up_scale), names[i]+'_T'+str(current_thresh)+ '.png')

    # load
    SR_in = cv2.imread(SR_in_pth)[:,:,(0,2,1)]
    HR_in = cv2.imread(HR_in_pth)[:,:,(0,2,1)]
    LR_in = cv2.imread(LR_in_pth)[:,:,(0,2,1)]
    Bic_in = cv2.imread(Bic_in_pth)[:,:,(0,2,1)]
    class_SR_in = cv2.imread(class_SR_in_pth, cv2.IMREAD_UNCHANGED) # load as greyscale
    # class_HR_in = cv2.imread(SR_in_pth, cv2.IMREAD_UNCHANGED)

# TODO: plot accuracy metrics, too
    axes2[0,i].imshow(HR_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor)
    axes2[1,i].imshow(LR_in[np.int(za[i]/up_scale):np.int(zb[i]/up_scale),np.int(za[i]/up_scale):np.int(zb[i]/up_scale):]*rescale_factor)
    axes2[2,i].imshow(Bic_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor) # <----- HERE: fix scaling for LR !
    axes2[3,i].imshow(SR_in[za[i]:zb[i],za[i]:zb[i]:]*rescale_factor, extent=(100.5, 150.5, 150.5,100.5)) # norm=
    axes2[4,i].imshow(class_SR_in[za[i]:zb[i],za[i]:zb[i]:], cmap='Greys')
    
    

    for j in range(5):
        # plot
        # fig=plt.figure()
        # fig.add_subplot(251)
        # axes[i,j].imshow(cv2.normalize(SR_in, norm_type=cv2.NORM_MINMAX))
        # axes[i,j].imshow(cv2.normalize(SR_in, None, 1.0, 0.0, cv2.NORM_L1))
        # axes.tick_params...
        # fig1.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
        
        # axes[j,i].axis('off')
        axes2[j,i].xaxis.set(ticks=(), ticklabels=())
        axes2[j,i].yaxis.set(ticks=(), ticklabels=())
        # plt.axis('off')
        if i==0:
            axes2[j,i].set(ylabel=ylabels[j])
        if j==0:
            axes2[j,i].set(title=labels[i]) # (title=names[i]
#%% plot and save
fig1.tight_layout()
fig2.tight_layout()
# fig1.show()
fig1.savefig('Fig_subsets.png', dpi=300)
fig2.savefig('Fig_subsets_zoom_out.png', dpi=300)
print('done')
    

# %%

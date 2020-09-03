#%%
#%matplotlib inline
import os
import sys
import cv2
import numpy as np
import getpass
import pickle
from scipy import stats as st
import matplotlib.pyplot as plt

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def view_max_vals():
    ''' modified from generate_mod_LR_bic.py.  Loop through files and figure out max value to see if dynamic range has been squashed.
    '''

    # set data dir
    if getpass.getuser()=='ethan_kyzivat' or getpass.getuser()=='ekaterina_lezine': # on GCP 
        sourcedir = '/data_dir/planet_sub_LR_cal/HR/x4'
    else: # other
        raise ValueError('input_folder not specified!')
        pass


    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)

    ## load hash
    f=open("cal_hash.pkl", "rb")
    hash=pickle.load(f)
    b=[3,2,4]

    # prepare data with augementation
    max=[]
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))

        # read image
        image = cv2.imread(os.path.join(sourcedir, filename), cv2.IMREAD_UNCHANGED) # apparently, this loads as 8-bit bit depth... Changed!

        # print max
        max.append(np.max(image))
        print('Max: {}'.format(max[i]))

    # save
    np.save('max.npy', max)

    # analyze

def plot_max(): # for analysis
    '''for analysis'''
    max=np.load('max.npy')
    stats=st.describe(max)
    print(stats)
    h=np.histogram(max)
    plt.show()
    plt.savefig('hist.png')
    pass

if __name__ == "__main__": # uncomment to swith between analysis and calculations
    view_max_vals() # calculate/save
    plot_max() # view/analyze


# %%

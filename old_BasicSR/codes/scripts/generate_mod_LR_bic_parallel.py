'''
Preprocessing: run 5th
'''
import os
import sys
import cv2
import numpy as np
import getpass
import pickle
from multiprocessing import Pool
import multiprocessing

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass

adjust_stretch=False # whether or not to change TDR in this step

def generate_mod_LR_bic(top_dir):
    # set parameters
    print(f'Top dir: {top_dir}')
    n_thread=multiprocessing.cpu_count() # num cores 8
    up_scale = 10
    mod_scale = 10
    stretch_multiplier=1 # to increase total dynamic range

    # set data dir
    if getpass.getuser()=='ethan_kyzivat' or getpass.getuser()=='ekaterina_lezine': # on GCP 
        sourcedir = os.path.join('/data_dir/planet_sub_v2', top_dir) # change HERE if updating
        savedir = os.path.join('/data_dir/', top_dir)
    else: # other
        raise ValueError('input_folder not specified!')
        pass

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)

    ## load hash
    #f=open("cal_hash.pkl", "rb")
    #hash=pickle.load(f)

    ## new parallel ##########################

    pool = Pool(n_thread)
    for i in range(num_files): # range(700): #
        filename = filepaths[i] # will this work?
        if os.path.isfile(saveHRpath+os.sep+filename)==False: # only write if file doesn't exist
            pool.apply_async(worker,
                            args=(i, filename, sourcedir, saveHRpath, saveLRpath, saveBicpath, up_scale, mod_scale, stretch_multiplier, hash)) # , callback=update
            # worker(i, filename, sourcedir, saveHRpath, saveLRpath, saveBicpath, up_scale, mod_scale, stretch_multiplier, hash) # for debugging
        else: # elif os.path.isfile(saveHRpath+os.sep+filename)==True: 
            print('Skip no. {}.'.format(i))
    pool.close()
    pool.join()
    print('All subprocesses done.')

    ## new parallel ##########################
    # prepare data with augementation
    ##
def worker(i, filename, sourcedir, saveHRpath, saveLRpath, saveBicpath, up_scale, mod_scale, stretch_multiplier,hash):

    # read image
    image = cv2.imread(os.path.join(sourcedir, filename), cv2.IMREAD_UNCHANGED) # apparently, this loads as 8-bit bit depth... Changed!

    if adjust_stretch:
        ## apply correction
        b=[3,2,4]
        image_cal=np.array(np.zeros(image.shape), dtype='double')
        ID=filename[:-10]
        coeffs=hash[ID]
        for j in range(3):
            image_cal[:,:,j]=image[:,:,j]*coeffs[b[j]]*255*stretch_multiplier
        image=image_cal.astype(np.uint8)

    ## continue
    width = int(np.floor(image.shape[1] / mod_scale))
    height = int(np.floor(image.shape[0] / mod_scale))
    # modcrop
    if len(image.shape) == 3:
        image_HR = image[0:mod_scale * height, 0:mod_scale * width, :] # this simply makes the dimenions of the image even if they weren't originally
    else:
        image_HR = image[0:mod_scale * height, 0:mod_scale * width]
    # LR
    image_LR = imresize_np(image_HR, 1 / up_scale, True)
    # bic
    image_Bic = imresize_np(image_LR, up_scale, True) # uses bicubic resampling to recreate the HR image from the LR naively (the GAN will do this better)

    cv2.imwrite(os.path.join(saveHRpath, filename), image_HR) 
    cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
    cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)
    print('No.{} -- Processed {}'.format(i, filename))
    return 'No.{} -- Processed {}'.format(i, filename)

if __name__ == "__main__":
    generate_mod_LR_bic('train_mod')
    generate_mod_LR_bic('valid_mod')
    generate_mod_LR_bic('hold_mod')
    # generate_mod_LR_bic('hold_mod_shield') # for shield scenes
    # generate_mod_LR_bic('hold_mod_shield_masks') # for shield scenes masks
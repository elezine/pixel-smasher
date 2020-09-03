
import pandas as pd
import pickle
import numpy as np

im_name = []
hr = []
lr = []
bic = []
sr = []

save_freq =

for image,x in enumerate(images):

    #classifer
    im_name.append(image)
    hr.append()
    lr.append()
    bic.append()
    sr.append()

    if x%save_freq == 0:
        l = list(zip(im_name, hr, lr, bic, sr))
        with open('checkpoint_' + str(x) + '.txt', 'wb') as fp:
            pickle.dump(l, fp)

df = pd.DataFrame(list(zip(im_name, hr, lr, bic, sr)), columns =['im','hr','lr','bic','sr'])
df.to_csv('df.csv')

'''
#if you want to open a saved pickle checkpoint:
with open("path", "rb") as fp:
    b = pickle.load(fp)
'''

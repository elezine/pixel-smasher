''' Breadcrumb trail for recording computations to determine mean histogram stretch for all images. Quite sloppy and slow bc of loop, but now modified to be better with a tiny aproximation.
Preprocessing: run 2nd

Input = 'codes/utils/histograms.npy'
OUtput = 'quantile_matrix.npy' '''
import numpy as np
from matplotlib import pyplot as plt

h=np.load('histograms.npy') # codes/utils/histograms.npy'
h_mean=np.mean(h, 2)
bands=['B','G','R','N']
quantiles=(0.2, 0.95)
# percentile_min=2
# percentile_max=95
    # plot
length=h_mean.shape[0]
if 1==0:
    f, ax = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        ax[i].set_title(f'Band {bands[i]}')
        ax[i].bar(range(l), h_mean[:,i])
    plt.show()
    pass

    # compute percentiles
if 1==0: # slooow way
    means=np.tile(np.arange(0.5, length+0.5, 1), (4,1)).T
    n_px=np.sum(h_mean, axis=0) # simulated pixels for mean image
    px=[]
    for i in range(4):
        px.append(np.zeros(0)) #np.zeros(len(n_px[i]))
        for j in range(len(h_mean)):
            px[i]=np.append(px[i], means[j,i]*np.ones(h_mean[j,i].round().astype(int)))

    np.quantile(px[0],quantiles)

    # fast, nearly precise way
bin_means=np.arange(0.5, length+0.5, 1)
h_mean_relative=h_mean/np.sum(h_mean, 0)
h_mean_cum=np.cumsum(h_mean_relative, 0)
quantile_val=np.zeros((2,4))
for i in range(4):
    for j in range(len(quantiles)):
        quantile_val[j,i]=bin_means[np.argmax(h_mean_cum[:,i][h_mean_cum[:,i]<= quantiles[j]]).round()]
print(f'Quantiles values:\n{quantile_val}')
np.save('quantile_matrix.npy', quantile_val)
pass
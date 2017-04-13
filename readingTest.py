import numpy as np
from numpy import genfromtxt
import scipy.io as sio
import csv

data=genfromtxt('/Users/luzh14/nilmtk/data/REDD/low_freq/house_1/channel_1.dat', delimiter=' ')
data=np.array(data)
save_fn = 'loadData.mat'
sio.savemat(save_fn, {'data': data.T})

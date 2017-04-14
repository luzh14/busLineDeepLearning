#-*- coding=utf8
import numpy as np
import scipy.io as sio
import pandas as pd
from pandas import Series
from timeStampInfo import timeStamp
from microwaveTimeStampInfo import microwaveTimeStamp
import datetime
import csv

channel_1=pd.read_csv('/Users/luzh14/nilmtk/data/REDD/low_freq/house_1/channel_1.dat', delimiter=' ')
channel_1=np.array(channel_1)


channel_11=pd.read_csv('/Users/luzh14/nilmtk/data/REDD/low_freq/house_1/channel_11.dat', delimiter=' ')
channel_11=np.array(channel_11)


main=Series(channel_1[:,1],index=[datetime.datetime.strptime(x,'%Y%m%d%H%M%S') for x in timeStamp])
main=main.resample('S',fill_method='ffill')[0:]#插值

microwave=Series(channel_11[:,1],index=[datetime.datetime.strptime(x,'%Y%m%d%H%M%S') for x in microwaveTimeStamp])
microwave=microwave.resample('S',fill_method='ffill')[0:]#插值

microPower=np.zeros(100,dtype=float)
mainInput=np.zeros((100,128),dtype=float)
for x in range(128,227):
    microPower[x-128]=microwave[x]
    print(x,'/',86400-128)
    for y in range(x-127,x):
        mainInput[x-128,y%127]=main[y]

save_fn = 'loadData.mat'
sio.savemat(save_fn, {'X':mainInput,'Y':microPower})


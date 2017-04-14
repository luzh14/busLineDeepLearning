#-*- coding=utf8
import numpy as np
from numpy import genfromtxt
import scipy.io as sio
import pandas as pd

import datetime


channel_11=pd.read_csv('/Users/luzh14/nilmtk/data/REDD/low_freq/house_1/channel_11.dat', delimiter=' ')
channel_11=np.array(channel_11)
date=channel_11[:,0]

a=[]
#把数据集中的时间戳转换为相应格式
for x in date:
    a.append(datetime.datetime.fromtimestamp(x).strftime('%Y%m%d%H%M%S'))


a=str(a)
f = open("microwaveTimeStampInfo.py",'wb')
f.write('microwaveTimeStamp=')
f.write(a)
f.close()

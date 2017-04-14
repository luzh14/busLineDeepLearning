import numpy as np
from numpy import genfromtxt
import scipy.io as sio
import pandas as pd
from pandas import Series
from redd import a

import datetime
import csv

channel_1=pd.read_csv('/Users/luzh14/nilmtk/data/REDD/low_freq/house_1/channel_1.dat', delimiter=' ')
channel_1=np.array(channel_1)
date=channel_1[:,0]

a=[]
for x in date:
    a.append(datetime.datetime.fromtimestamp(x).strftime('%Y%m%d%H%M%S'))


a=str(a)
f = open("a.txt",'wb')
f.write('a=')
f.write(a)
f.close()

ts=Series(channel_1[:,1],index=[datetime.datetime.strptime(x,'%Y%m%d%H%M%S') for x in a])

print(ts)
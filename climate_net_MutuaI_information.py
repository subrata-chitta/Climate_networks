#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:04:33 2023

@author: cssc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:22:31 2023

@author: cssc
"""
#import numpy as np
#import xarray as xr
#import netCDF4 as nc
#import pandas as pd
#import matplotlib.pyplot as plt
#import os
#import glob
#from scipy import stats
#import scipy.io
#import warnings
#warnings.filterwarnings("ignore")
##import pyunicorn
##import cdo
##import requests
#path = '/home/cssc/Desktop/Data/Auroop Ganguly/climate_tutorial/'  
#folder = 'climate_network/'
### load data file names    
#all_files = glob.glob(path + folder+'*.nc')
#all_files.sort()
## Create list for 
#individual_files = []
#
#for i in all_files:
#    timestep_ds = xr.open_dataset(i)    
#    individual_files.append(timestep_ds)
#
#new_ds = xr.concat(individual_files, dim='time')
#print(new_ds)
#
#all_data=[]
#for i in range(0,len(new_ds["precip"]["lat"])):
#    all_data.append([])
#    for j in range(0,len(new_ds["precip"]["lon"])):
#        all_data[i].append([])
#        longitude = new_ds["precip"]["lon"].values[j]
#        latitude = new_ds["precip"]["lat"].values[i]
#        one_point = new_ds["precip"].sel(lat=latitude, lon=longitude)
#        df = one_point.to_dataframe()
#        df_prec=df.precip
#        all_data[i][j].append(df_prec)
#        
#a=np.zeros([len(all_data)*len(all_data[0]),len(df)],float)    
#for k in range(len(all_data)):
#    for l in range(len(all_data[0])):
#        for j in range(0,len(df)):
#            a[300*k+l,j]= all_data[k][l][0][j]
#
#
#df = pd.DataFrame(a)
#df1=df.dropna()
#len(df1[0])
#plt.plot(df1[0],'b.')
#df2=df1.to_numpy()
#####################################################################################
##Data
#scipy.io.savemat('usa_check_4.mat', {'prec': df2}) 
#####################################################################################
import scipy.io
import numpy as np

mat = scipy.io.loadmat('usa_check_4')
df_main = np.array(mat['prec'])

histogram = np.histogram
histogram2d = np.histogram2d
log = np.log
#self.data.normalize_time_series_array(anomaly)
n_samples = len(df_main)
mi = np.zeros((len(df_main), len(df_main)))  #initialize MI
#  Get common range for all histograms
range_min = df_main.min()
range_max = df_main.max()
#  Calculate the histograms for each time series
n_bins=50
p = np.zeros((len(df_main), n_bins))
for i in range(len(df_main)):
    p[i, :] = histogram(df_main[i,:], bins=n_bins, range=(range_min, range_max))[0].astype("float64")
#  Normalize by total number of samples = length of each time series
p /= n_samples
#  Make sure that bins with zero estimated probability are not counted in the entropy measures.
p[p == 0] = 1
#  Compute the information entropies of each time series
H = - (p * log(p)).sum(axis=1)
#  Calculate only the lower half of the MI matrix, since MI is symmetric with respect to X and Y.
        
#  Calculate the joint probability distribution#  Calculate the joint probability distribution
for i in range(len(df_main)):
    for j in range(i):
        pxy = histogram2d(df_main[i,:], df_main[j,:], bins=n_bins, range=((range_min, range_max), (range_min, range_max)))[0].astype("float64")
        #  Normalize joint distribution
        pxy /= n_samples
#  Compute the joint information entropy
        pxy[pxy == 0] = 1
        HXY = - (pxy * log(pxy)).sum()
                
        mi.itemset((i, j), H.item(i) + H.item(j) - HXY)
        mi.itemset((j, i), mi.item((i, j)))

np.savetxt('adj_mutual_info.txt',mi) 

import time
end=time.time()
print(end)



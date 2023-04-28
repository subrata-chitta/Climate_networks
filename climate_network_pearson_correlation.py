#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:22:31 2023

@author: cssc
"""
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats
import scipy.io
import warnings
warnings.filterwarnings("ignore")
#import pyunicorn
#import cdo
#import requests
path = '/home/cssc/Desktop/Data/Auroop Ganguly/climate_tutorial/'  
folder = 'climate_network/'
## load data file names    
all_files = glob.glob(path + folder+'*.nc')
all_files.sort()
# Create list for 
individual_files = []

for i in all_files:
    timestep_ds = xr.open_dataset(i)    
    individual_files.append(timestep_ds)

new_ds = xr.concat(individual_files, dim='time')
print(new_ds)

new_ds.info()
new_ds.nbytes/1e9
new_ds.variables

# daily mean precipitation over the region
mean_prec = new_ds.mean(dim='lat')
mean_prec = mean_prec.mean(dim = 'lon')
print(mean_prec)
df = mean_prec.to_dataframe()
df.plot()
plt.show()
# or

new_ds.precip.mean(dim=['lat', 'lon']).plot();
# heat map of precipitation over the region of a particular data point
precip = new_ds.sel(time = '2013-06-04')
print(precip)
precip['precip'].plot()
plt.show()

# time series of particular one grid point
key1=70 
key2=70 
longitude = new_ds["precip"]["lon"].values[key1]
latitude = new_ds["precip"]["lat"].values[key2]
new_ds.precip.sel(lat=latitude, lon=longitude).plot();

# daily mean precipitation  over US
mean_precip = new_ds.mean(dim='lat')
mean_precip = mean_precip.mean(dim = 'lon')
#print(mean_precip)
# Yearly mean precipitation over US (CONUS) 
annual_mean  = mean_precip.groupby('time.year').mean('time')
print(annual_mean)
df1 = annual_mean.to_dataframe()
df1.plot()
plt.show()
df1.to_csv('mean_annual_precipitation.csv')
# linear fit
coefficients, residuals, _, _, _ = np.polyfit(range(len(df1.index)),df1,1,full=True)
print('Slope ' + str(coefficients[0]))
print('Intercept ' + str(coefficients[1]))
x = range(2013, 2022+1)
y = ([coefficients[0]*x + coefficients[1] for x in range(len(df1))])
plt.plot(df1)
plt.plot(x,y)

# Yearly max precipitation (mean over region) over US (CONUS) 
annual_max  = mean_precip.groupby('time.year').max('time')
print(annual_max)
df2 = annual_max.to_dataframe()
df2.plot()
plt.show()
df2.to_csv('maximum_annual_precipitation.csv')
# linear fit
coefficients, residuals, _, _, _ = np.polyfit(range(len(df2.index)),df2,1,full=True)
print('Slope ' + str(coefficients[0]))
print('Intercept ' + str(coefficients[1]))
x = range(2013, 2022+1)
y = ([coefficients[0]*x + coefficients[1] for x in range(len(df2))])
plt.plot(df2)
plt.plot(x,y)

all_data=[]
for i in range(0,len(new_ds["precip"]["lat"])):
    all_data.append([])
    for j in range(0,len(new_ds["precip"]["lon"])):
        all_data[i].append([])
        longitude = new_ds["precip"]["lon"].values[j]
        latitude = new_ds["precip"]["lat"].values[i]
        one_point = new_ds["precip"].sel(lat=latitude, lon=longitude)
        df = one_point.to_dataframe()
        df_prec=df.precip
        all_data[i][j].append(df_prec)
        
a=np.zeros([len(all_data)*len(all_data[0]),len(df)],float)    
for k in range(len(all_data)):
    for l in range(len(all_data[0])):
        for j in range(0,len(df)):
            a[300*k+l,j]= all_data[k][l][0][j]

import scipy.io
scipy.io.savemat('prec_usa.mat', {'prec': a}) 

mat = scipy.io.loadmat('prec_usa')
rain = np.array(mat['prec'])


df = pd.DataFrame(rain)
df1=df.dropna()
len(df1[0])
plt.plot(df1[0],'b.')
df2=df1.to_numpy()

A1=np.zeros([len(df2),len(df2)],float)
A2=np.zeros([len(df2),len(df2)],float)

for i in range(0, len(df2)):
    print(i)
    for j in range(0, len(df2)):
        A1[i,j]=stats.pearsonr(df2[i,:],df2[j,:])[0]
#        A[i,j]=stats.pearsonr(df2.iloc,df2.iloc[j]).statistic
  
np.savetxt('adj_pearson.txt',A1) 



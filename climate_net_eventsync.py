#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:54:59 2023

@author: cssc
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('usa_check_4')
df_main = np.array(mat['prec'])


B=np.zeros( [len(df_main[:,0]), len(df_main[0,:]) ], float)
C=[]
for j in range(len(df_main[:,0])):
    C.append([])
    for i in range(len(df_main[0,:])-1):
        if( ( df_main[j,i] > df_main[j,i-1] ) and ( df_main[j,i] > df_main[j,i+1] )):
            B[j, i]=1
            C[j].append(i)
        else:
            B[j, i]=0

result=np.zeros([len(df_main[:,0]), len(df_main[:,0])], float)
for m in range(len(df_main[:,0])):
    print(m)
    for n in range(len(df_main[:,0])):
        ex=C[m]
        ey=C[n]
# Dynamical delay
        diffx = np.diff(ex)
        diffy = np.diff(ey)
        diffxmin = np.minimum(diffx[ 1:], diffx[ :-1])
        diffymin = np.minimum(diffy[ 1:], diffy[ :-1])

        lag = 0.5*min(min(diffx),min(diffy),min(diffxmin),min(diffymin))
#lag=10

#plt.plot(df_main[0,:],'ko') 
#plt.plot(df_main[5,:],'m.')

        Jxy=np.zeros( [len(C[m]), len(C[n]) ], float)

        for i in range(0, len(C[m]) ):
            for j in range(0, len(C[n]) ):
                if ( (C[m][i]-C[n][j])==0):
                    Jxy[i,j]=0.5
                elif( 0 < (C[m][i]-C[n][j]) <= lag): 
                    Jxy[i,j]=1.0
                else: 
                    Jxy[i,j]=0.0
            
                Cxy=np.sum(np.sum(Jxy))            
            
        Jyx=np.zeros( [len(C[n]), len(C[m]) ], float)

        for i in range(0, len(C[n]) ):
            for j in range(0, len(C[m]) ):
                
                if ( (C[n][i]-C[m][j])==0):
                    Jyx[i,j]=0.5
                elif( 0 < (C[n][i]-C[m][j]) <= lag): 
                    Jyx[i,j]=1.0
                else: 
                    Jyx[i,j]=0.0
                    
        Cyx=np.sum(np.sum(Jyx)) 

        Qtau_1=(Cxy+Cyx)/(np.sqrt((len(C[m]) ) * (len(C[n]) )) )
        Qtau_2=(Cyx-Cxy)/(np.sqrt((len(C[m]) ) * (len(C[n]) )) )
        
        result[m,n]=Qtau_1

np.savetxt('adj_event_sync.txt',result) 
#plt.plot(BC, 'k.')  
#plt.plot(df_main[0,:]/max(df_main[0,:]), 'r.')          
        
    
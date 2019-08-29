#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 09:24:11 2018

@author: zjr
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt  
import os




def draw_picture(prediction,test_y):
       
#    prediction=np.reshape(prediction,(-1,1))
#    test_y=np.reshape(test_y,(-1,1))
#    
#    print("mean_absolute_error is:",mean_absolute_error (test_y,prediction))
#
#    print("mean_squared_error is:",mean_squared_error (test_y,prediction))
#
#    print("the average MAPE is :",MAPE(prediction,test_y))
    
    
    preandtest={'test':test_y,'prediction':prediction}
        
    dataframe=pd.DataFrame(preandtest,columns=['test','prediction'])
    
    if os.path.isfile('/home/user/zjr/volity/TCN_GRU/data.csv'):
        dataframe.to_csv('/home/user/zjr/volity/TCN_GRU/data.csv',mode='a',header=False,index=False)
    else:
        dataframe.to_csv('/home/user/zjr/volity/TCN_GRU/data.csv',index=False)
    
    print("Data has already been saved!")

#    fig = plt.figure(figsize=(10,5))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80  
#    axes = fig.add_subplot(1, 1, 1)    
#       
#    line1,=axes.plot(range(len(prediction)), prediction, 'r--',label='predict',linewidth=2)  
#    line2,=axes.plot(range(len(test_y)), test_y, 'b',label='actual',linewidth=2)    
#    
#    axes.grid()  
#    fig.tight_layout()  
#    plt.legend(handles=[line1,line2])  
#    plt.title('nodatatwo')  
#    plt.show()
  

def MAPE(prediction,test_y):
    maps=[]
    for i in range(len(prediction)):
        distance=prediction[i]-test_y[i]
        mape=distance/prediction[i]
        if mape <=0:
            mape=-mape
        maps.append(mape)
    results=0
    for j in maps:
        results=results+j
        
    lastresults=results/len(prediction)
    
    return lastresults


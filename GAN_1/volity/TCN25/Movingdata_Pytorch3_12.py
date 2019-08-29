#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:05:58 2018

@author: zjr
"""

import torch
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine 
import matplotlib.pyplot as plt
from sklearn import preprocessing  
from torch.autograd import Variable
pymysql.install_as_MySQLdb()

#torch.cuda.set_device(1)

def get_index_data(code):

    name = 'root'
    password = 'szU@654321'
    db_name = 'stock_db'
    db_ip = '210.39.12.25'
    db_port = '50002'
   
    engine = create_engine('mysql://' + name + ':' + password
                           + '@' + db_ip +':'+db_port+ '/' + db_name + '?charset=utf8')
    
    table_name = 'index_data_' + code
    
    sql = 'SELECT date, high, low, open ,close , pctchange from ' + table_name + ' ORDER BY date'                          
    df1 = pd.read_sql(sql, engine)
    if len(df1) > 0:
        
        df1.rename(columns={'high': 'highprice'}, inplace=True)
        df1.rename(columns={'low': 'lowprice'}, inplace=True)
        df1.rename(columns={'open':'openprice'},inplace=True)
        df1.rename(columns={'close':'closeprice'},inplace=True)
        df1.rename(columns={'pctchange':'pctchange'},inplace=True)

    
    return df1


#get label and data
#original_data,FLAGS.channel,FLAGS.batch_size, FLAGS.train_start ,FLAGS.test_end, FLAGS.train_end
#    original_data,channel=1,batch_size=64,train_start=0 ,test_end=1501, train_end=1500
def get_all_data(original_data,input_num,test_day,train_start ,test_end, train_end,train):
    
    
    
    
    a={"highprice":0,"lowprice":0,"closeprice":0,"pctchange":0,"openprice":0,"date":0,}   
    original_data=original_data.append(a,ignore_index=True)
    
#    original_data=get_index_data('000001')

#processed---close ,not processed-----unpre_close
#-------------------------------------------------  
    highprice=original_data.iloc[:,1]
    lowprice=original_data.iloc[:,2]
    openprice=original_data.iloc[:,3]
    closeprice=original_data.iloc[:,4]
    pctchange=original_data.iloc[:,5]
    unpre_close=original_data.iloc[:,4] 
    

    
    
    
#    highprice=highprice.append(last_num)
#    highprice=np.array(highprice,np.float64)
#    closeprice=np.array(closeprice,np.float64)
    
    
#total data  
#------------------------------------------------- 
        
    close=closeprice[train_start:test_end+test_day]
    high=highprice[train_start:test_end+test_day]
    low=lowprice[train_start:test_end+test_day]
    opens=openprice[train_start:test_end+test_day]
    pctchange=pctchange[train_start+1:test_end+1+test_day]
    unpre_close=unpre_close[train_start:test_end+test_day]
    
   
  
#process
#-------------------------------------------------      
    close=process(close)
    high=process(high)
    low=process(low) 
    opens=process(opens)
    
 
    
#smooth average
#-------------------------------------------------      
    smoothed_close=exponential_smoothing(close,alpha=0.8)
    high =exponential_smoothing(high,alpha=0.8)
    opens =exponential_smoothing(opens,alpha=0.8)
    low=exponential_smoothing(low,alpha=0.8)  
    pctchange=exponential_smoothing(pctchange,alpha=0.8)  
    smomothed_test=exponential_smoothing(close,alpha=0.8)
    
#reshape  
#------------------------------------------------- 
    close=np.reshape(np.array(smoothed_close,dtype=np.float32),(-1,1))
    unpre_close=np.reshape(np.array(unpre_close,dtype=np.float32),(-1,1))
    high=np.reshape(np.array(high,dtype=np.float32),(-1,1))
    opens=np.reshape(np.array(opens,dtype=np.float32),(-1,1))
    low=np.reshape(np.array(low,dtype=np.float32),(-1,1))
    pctchange=np.reshape(np.array(pctchange,dtype=np.float32),(-1,1))
    

  

#get train data and test data  and label  
#-------------------------------------------------      

    close_train_data=close[0:train_end-train_start]
    close_test_data=close[train_end-train_start-input_num-1:train_end-train_start+test_day] 
    
    high_train_data=high[0:train_end-train_start]
    high_test_data=high[train_end-train_start-input_num-1:train_end-train_start+test_day]  
    
    open_train_data=opens[0:train_end-train_start]
    open_test_data=opens[train_end-train_start-input_num-1:train_end-train_start+test_day] 
    
    low_train_data=low[0:train_end-train_start]
    low_test_data=low[train_end-train_start-input_num-1:train_end-train_start+test_day]  
    
    pct_train_data=pctchange[0:train_end-train_start]
    pct_test_data=pctchange[train_end-train_start-input_num-1:train_end-train_start+test_day]
    
    
    
#you can set  label here   
#-------------------------------------------------      
#-------------------------------------------------  

    
    train_data= close_train_data
    test_data=  close_test_data
    
    
    
    train_label=get_train_label(train_start,train_end,train_data,input_num)
    test_label=get_test_label(train_end,test_end,test_data,test_day)  
    
        
    reallabel=unpre_close[len(train_data)+1:]   
    delta=unpre_close[len(train_data):-1]
    smoothed_data_pre=smoothed_close[len(train_data)-1:-1]   
    

    
   
#------------------------------------------------------------------------------------------        
#------------------------------------------------- -----------------------------------------           
#    fig = plt.figure(figsize=(10,6))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80  
#    axes = fig.add_subplot(1, 1, 1)    
#       
#    line1,=axes.plot(range(len(close)), close,label='Processed close price',linewidth=1)      
#    line3,=axes.plot(range(len(nclose)), nclose,label='Normalized close price',linewidth=1)    
#       
#    axes.grid()  
#    fig.tight_layout()  
#    plt.legend(handles=[line1,line3])  
#    plt.title('Normalize')  
#    
#------------------------------------------------------------------------------------------        

    
#divide data 
#-------------------------------------------------         
  
    close_train_data_one,close_train_data_two = get_data(test_day,train_end,train_start,close_train_data,input_num,train=True)
    close_test_data_one,close_test_data_two = get_data(test_day,test_end,train_end,close_test_data,input_num,train=False)   
  

    high_train_data_one,high_train_data_two = get_data(test_day,train_end,train_start,high_train_data,input_num,train=True)   
    high_test_data_one,high_test_data_two = get_data(test_day,test_end,train_end,high_test_data,input_num,train=False)   

    low_train_data_one,low_train_data_two = get_data(test_day,train_end,train_start,low_train_data,input_num,train=True)
    low_test_data_one,low_test_data_two= get_data(test_day,test_end,train_end,low_test_data,input_num,train=False)    
    
    open_train_data_one,open_train_data_two = get_data(test_day,train_end,train_start,open_train_data,input_num,train=True)
    open_test_data_one,open_test_data_two = get_data(test_day,test_end,train_end,open_test_data,input_num,train=False)   
    
    pct_train_data_one,pct_train_data_two = get_data(test_day,train_end,train_start,pct_train_data,input_num,train=True)
    pct_test_data_one,pct_test_data_two = get_data(test_day,test_end,train_end,pct_test_data,input_num,train=False)   
   
  
       

#    close_train_data_one=np.array(close_train_data_one,np.float32)
#    close_train_data_two=np.array(close_train_data_two,np.float32)   

#normalize   
#------------------------------------------------------------------------------------------       
#------------------------------------------------------------------------------------------
    ss_y= preprocessing.MinMaxScaler() 
    ss_close = preprocessing.MinMaxScaler()   
    ss_high = preprocessing. MinMaxScaler()
    ss_open = preprocessing. MinMaxScaler() 
    ss_low = preprocessing. MinMaxScaler()  
    ss_pct= preprocessing. MinMaxScaler()
     
#train data    
    close_train_data_one=ss_close.fit_transform(close_train_data_one)
    high_train_data_one=ss_high.fit_transform(high_train_data_one)
    open_train_data_one=ss_open.fit_transform(open_train_data_one)
    low_train_data_one=ss_low.fit_transform(low_train_data_one)    
    pct_train_data_one=ss_pct.fit_transform(pct_train_data_one)
    
    close_train_data_two=ss_close.fit_transform(close_train_data_two)
    high_train_data_two=ss_high.fit_transform(high_train_data_two)
    open_train_data_two=ss_open.fit_transform(open_train_data_two)
    low_train_data_two=ss_low.fit_transform(low_train_data_two)    
    pct_train_data_two=ss_pct.fit_transform(pct_train_data_two)
    
#test data   
    
 
    close_test_data_one=ss_close.transform(close_test_data_one)
    high_test_data_one=ss_high.transform(high_test_data_one)
    open_test_data_one=ss_open.transform(open_test_data_one)
    low_test_data_one=ss_low.transform(low_test_data_one)    
    pct_test_data_one=ss_pct.transform(pct_test_data_one)
    
    close_test_data_two=ss_close.transform(close_test_data_two)
    high_test_data_two=ss_high.transform(high_test_data_two)
    open_test_data_two=ss_open.transform(open_test_data_two)
    low_test_data_two=ss_low.transform(low_test_data_two)    
    pct_test_data_two=ss_pct.transform(pct_test_data_two)




    train_label=np.reshape(train_label,(-1,1))
    test_label=np.reshape(test_label,(-1,1))
    
    
    train_label=ss_y.fit_transform(train_label)
    test_label=ss_y.transform(test_label)  



    
   
#-------------------------------------------------  
#-------------------------------------------------  
    train_data_one=convert_data(close_train_data_one,high_train_data_one,low_train_data_one,open_train_data_one,pct_train_data_one,high_train_data_one,low_train_data_one,open_train_data_one,input_size=input_num)
    train_data_two=convert_data(close_train_data_two,high_train_data_two,low_train_data_two,open_train_data_two,pct_train_data_two,high_train_data_two,low_train_data_two,open_train_data_two,input_size=1)
    test_data_one=convert_data(close_test_data_one,high_test_data_one,low_test_data_one,open_test_data_one,pct_test_data_one,high_test_data_one,low_test_data_one,open_test_data_one,input_size=input_num)
    test_data_two=convert_data(close_test_data_two,high_test_data_two,low_test_data_two,open_test_data_two,pct_test_data_two,high_test_data_two,low_test_data_two,open_test_data_two,input_size=1)
  
    
#-------------------------------------------------
#for pytorch
#-------------------------------------------------  


    train_data_one=np.transpose(train_data_one,(0,2,1))
    train_data_two=np.transpose(train_data_two,(0,2,1))   
    test_data_one=np.transpose(test_data_one,(0,2,1)) 
    test_data_two=np.transpose(test_data_two,(0,2,1)) 
  
    
    X_train=torch.from_numpy(np.concatenate((train_data_one,train_data_two),axis=2))
    X_test=torch.from_numpy(np.concatenate((test_data_one,test_data_two),axis=2))  
    Y_train=torch.from_numpy(np.array(train_label,np.float64))
    Y_test=torch.from_numpy(np.array(test_label,np.float64))
    
    
    

    
    
    
    X_train=X_train.type(torch.FloatTensor)
    X_test=X_test.type(torch.FloatTensor)
    Y_train=Y_train.type(torch.FloatTensor)
    Y_test=Y_test.type(torch.FloatTensor)
            
#    X_train=X_train.permute(0,2,1)
#    X_test=X_test.permute(0,2,1)

    
    
    return X_train,X_test,Y_train,Y_test,reallabel,delta,smoothed_data_pre,ss_y
    

           
    

#def convert_data(data_one,data_two,data_three,data_four,data_five,input_size):
#    i=0
#
#    
#    last=[]
#    data_one=list(data_one)
#    data_two=list(data_two)
#    data_three=list(data_three)
#    data_four=list(data_four)
#    data_five=list(data_five)
#   
#    while i < len(data_one):
#        pre=[]
#        pre.append(data_one[i:i+1]+data_two[i:i+1]+
#                   data_three[i:i+1]+data_four[i:i+1]+data_five[i:i+1])
#  
#
#        last.append(pre)     
#        i=i+1        
#          
#    lasts=np.reshape(last,(-1,5,input_size))
#    lasts=np.transpose(lasts,(0,2,1))
#    
#    return lasts


def convert_data(data_one,data_two,data_three,data_four,data_five,data_six,data_seven,data_eight,input_size):
    i=0

    
    last=[]
    data_one=list(data_one)
    data_two=list(data_two)
    data_three=list(data_three)
    data_four=list(data_four)
    data_five=list(data_five)
    data_six=list(data_six)
    data_seven=list(data_seven)
    data_eight=list(data_eight)
    
   
    while i < len(data_one):
        pre=[]
        pre.append(data_one[i:i+1]+data_two[i:i+1]+
                   data_three[i:i+1]+data_four[i:i+1]+data_five[i:i+1]+
                   data_six[i:i+1]+data_seven[i:i+1]+data_eight[i:i+1]
                   
                   )
  

        last.append(pre)     
        i=i+1        
          
    lasts=np.reshape(last,(-1,8,input_size))
    lasts=np.transpose(lasts,(0,2,1))
    
    return lasts





def exponential_smoothing(data,alpha):
    
    data=np.array(data,np.float32)
    s2=np.zeros(data.shape) 
    s2[0]=data[0]    
    
    for i in range(1,len(s2)):
        s2[i]=alpha*data[i]+(1-alpha)*s2[i-1]
        
    return s2




#caculate data[i]-data[i-1]|/data[i-1]
def process(price):
    datas=[]
    

    unpreprice=price
    price=price[1:len(price)]
    price=list(price) 
    
    price_pre=unpreprice
    price_pre=price_pre[0:len(price_pre)-1]
    price_pre=list(price_pre)
    
#the length of closeprice and closeprice_pre should be the same
    for i in range(len(price_pre)):
        data=(price[i]-price_pre[i])/price_pre[i]
        datas.append(data)
    for i in range(len(datas)):
        if datas[i] <=-0.2 :
           datas[i]=-0.2
        if datas[i] >=0.2:
           datas[i]=0.2
            
    return datas
    

   
    

    
    

def get_data(test_day,end,start,data,input_num,train):

    data_one=[]
    data_two=[]

   
    if train is True:
        for i in range(end-start):            
            data_one.append(data[i:i+input_num])
            data_two.append(data[i+input_num:i+input_num+1])         
            if data[i+input_num+1:i+input_num+2]==data[-1]:
                break
            
    else:
        for j in range(end-start+test_day-1):
            data_one.append(data[j:j+input_num])
            data_two.append(data[j+input_num:j+input_num+1])
            

    data_one=np.reshape(np.array(data_one,np.float64),(-1,input_num))
    data_two=np.reshape(np.array(data_two,np.float64),(-1,1))
    return data_one,data_two
    




          
def get_train_label(start,end,data,input_num):
    
    label=[]
    
    
    for i in range(end-start):
        label.append(data[i+input_num+1:i+input_num+2])   
        if data[i+input_num+1:i+input_num+2] == data[-1]:
            break   
    return label   
    





def get_test_label(start,end,data,test_day):
    label=[]
    label=data[-test_day:]
    return label
    
    
    
    


def normalized(data):
    ndata=[]
  
    data_min=np.min(data)
    data_max=np.max(data)    
    for i in range(len(data)):
        x=(data[i]-data_min)/(data_max-data_min)
        ndata.append(x)
        
    return data_min,data_max,ndata
        
    

    
    
#original_data=get_index_data('000001')
#X_train,X_test,Y_train,Y_test,reallabel,delta,smoothed_data_pre,ss_y=get_all_data(
#        original_data,input_num=100,test_day=1,train_start=0,test_end=4643, train_end=4642,train=True)





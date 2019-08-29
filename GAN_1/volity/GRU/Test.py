#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:19:33 2019

@author: zjr
"""

import torch
import pandas as pd
from sqlalchemy import create_engine 
import warnings
import Save_Data
import numpy as np
from TrainandTest import TrainandTest
from  Modeltype import Modeltype

torch.cuda.set_device(5)

warnings.filterwarnings("ignore")    
 

class RunExperiment:
    def __init__(self, location,batch_size, cuda, droupout, clip,epochs,ksize,levels,log_interval,lr,optims,nhid,seed,
                 predictday,input_channels,n_classes,hidden_size,num_layers,input_num,test_day,train_start,test_end,train_end,code,modeltype):
        
        self.batch_size=batch_size
        self.cuda=cuda
        self.droupout=droupout
        self.clip = clip

        self.epochs=epochs
        self.ksize = ksize
        self.levels=levels
        self.log_interval=log_interval
        self.lr=lr
        self.optims=optims
        self.nhid=nhid
        self.seed=seed
        self.location=location

        #-----------------------

     
        self.predictday=predictday
        self.input_channels=input_channels
        self.n_classes=n_classes
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.input_num=input_num
        self.test_day=test_day
        self.train_start=train_start
        self.test_end=test_end
        self.train_end=train_end
        
       #-----------------------
       
        self.channel_sizes = [nhid]*levels
        self.code=code
        self.modeltype=modeltype
       #-----------------------
               
        self.original_data=self.Getpd(self.location)
        
       #-----------------------
        self.modeltp=Modeltype(self.modeltype,self.input_channels,self.n_classes,self.channel_sizes,self.ksize, self.droupout,
                               self.input_num,self.hidden_size,self.num_layers)
        
        self.model=self.modeltp.Models(self.modeltype,self.input_channels,self.n_classes,self.channel_sizes,self.ksize,self.droupout,
                             self.input_num,self.hidden_size,self.num_layers)
        
           
        self.traintest=TrainandTest(self.epochs,self.model,self.original_data,self.batch_size,self.lr,self.input_num,
                                 self.test_day,self.train_start ,self.test_end,self.train_end,self.cuda,self.optims,
                                 self.clip,self.log_interval)

       
    def Getpd(self,location):
        
        date=pd.read_csv(location,usecols=[0])   
        openprice = pd.read_csv(location,usecols=[1])   
        highprice=pd.read_csv(location,usecols=[2])   
        lowprice=pd.read_csv(location,usecols=[3])   
        closeprice=pd.read_csv(location,usecols=[24])     
        pctchange=pd.read_csv(location,usecols=[18])     
        date=list(date.iloc[:,0])
        openprice=list(openprice.iloc[:,0])
        highprice=list(highprice.iloc[:,0])
        lowprice=list(lowprice.iloc[:,0])
        closeprice=list(closeprice.iloc[:,0]) 
        pctchange=pctchange.iloc[:,0]  
        df={'a':date,'b':highprice,'c':lowprice,'d':openprice,'e':closeprice,'f':pctchange}
        original_data=pd.DataFrame(df)
        return original_data
    
#    def get_index_data(self,code):
#        name = 'root'
#        password = 'szU@654321'
#        db_name = 'stock_db'
#        db_ip = '210.39.12.25'
#        db_port = '50002' 
#        engine = create_engine('mysql://' + name + ':' + password
#                           + '@' + db_ip +':'+db_port+ '/' + db_name + '?charset=utf8')
#        table_name = 'index_data_' + code
#    
#        sql = 'SELECT date, high, low, open ,close , pctchange from ' + table_name + ' ORDER BY date'                          
#        df1 = pd.read_sql(sql, engine)
#        if len(df1) > 0:
#            df1.rename(columns={'high': 'highprice'}, inplace=True)
#            df1.rename(columns={'low': 'lowprice'}, inplace=True)
#            df1.rename(columns={'open':'openprice'},inplace=True)
#            df1.rename(columns={'close':'closeprice'},inplace=True)
#            df1.rename(columns={'pctchange':'pctchange'},inplace=True)
#        return df1



if __name__ == '__main__':
    
    originallabel=[]
    predictlabel=[]
    
    location='/home/tom/zjr/volity/GRU/000001.csv'
    
    modeltype='GRU'#'Wavenet','TCN','TCN_BLSTM','BLSTM'
    levels=4
    epochs=130
    predictday=350
    
    
    
    input_channels=8
    input_num=117
    
    
    batch_size=32     
    hidden_size=25
    train_start=0
    test_end=4148
    train_end=4147
    droupout=0.0 

  
    cuda='store_false'  
    clip =-1    
    ksize = 7    
    log_interval=131
    lr=0.0001
    optims = 'Adam'
    nhid=25
    seed=1111
    n_classes=1 
    num_layers=1 
    test_day=1
    
  
    
    channel_sizes = [nhid]*levels 
    code='000001'
    
       
    
    
    
    experiment = RunExperiment(location,batch_size, cuda, droupout, clip,epochs,ksize,levels,log_interval,lr,optims,nhid,seed,
                               predictday,input_channels,n_classes,hidden_size,num_layers,input_num,test_day,train_start,test_end,train_end,code,modeltype)

  
    
#    original_data=experiment.get_index_data(code)
    original_data=experiment.Getpd(location)
    
#------------------------------------------------------------------------------------------------------------- 
#Predict day
#------------------------------------------------------------------------------------------------------------- 
    for i in range (predictday): 
        
        print ("Predict day",i+1)
        
        print("Model type",modeltype)
        
        print ("------------------------")
        
        model=experiment.modeltp.Models(modeltype,input_channels,n_classes,channel_sizes,ksize, droupout,
                             input_num,hidden_size,num_layers)
        model=model.cuda()
        
        
        predict,reallabel=experiment.traintest.train_evaluate(epochs,model,original_data,batch_size,lr,input_num,test_day,train_start,test_end,train_end,
                                    cuda,optims,clip,log_interval)
        
        
         
        predict=np.reshape(np.array(predict,np.float32),(-1,))
        predict=list(predict)     
        
        reallabel=np.reshape(np.array(reallabel,np.float32),(-1,))
        reallabel=list(reallabel)
        
        Save_Data.draw_picture(predict,reallabel)
       
              
        train_end=train_end+1
        test_end=test_end+1
    
 
    
#-------------------------------------------------------------------------------------------------------------     

#------------------------------------------------------------------------------------------------------------- 
#Validation
#------------------------------------------------------------------------------------------------------------- 
#    for i in range (predictday): 
#        
#        print ("Predict day",i+1)
#        
#        print("Model type",modeltype)
#        
#        print ("------------------------")
#        
#        model=experiment.modeltp.Models(modeltype,input_channels,n_classes,channel_sizes,ksize, droupout,
#                             input_num,hidden_size,num_layers)
#        
#        current_loss,validation_loss=experiment.traintest.validation(epochs,model,original_data,batch_size,lr,input_num,test_day,train_start,test_end,train_end,
#                                    cuda,optims,clip,log_interval)
#        
#    
#        Draw_validation_loss.draw_picture(current_loss,validation_loss)
#    
#        
#        train_end=train_end+1
#        test_end=test_end+1








#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:23:53 2019

@author: zjr
"""



import wavenetmodel
import models
import tcnblstmmodel
import Blstm
import LSTM
import GRU
import warnings
warnings.filterwarnings("ignore")  

class Modeltype:
    
    def __init__(self,modeltype,input_channels, n_classes, channel_sizes, kernel_size, dropout,
                 input_num,hidden_size,num_layers): 
        self.modeltype=modeltype
        self.input_channels=input_channels
        self.n_classes=n_classes
        self.channel_sizes=channel_sizes
        self.kernel_size=kernel_size
        self.dropout=dropout
        self.input_num=input_num
        self.hidden_size=hidden_size
        self.num_layers=num_layers     
        #---------------------    
        self.model=self.Models(self.modeltype,self.input_channels,self.n_classes,
                               self.channel_sizes,self.kernel_size,self.dropout,
                               self.input_num,self.hidden_size,self.num_layers)
        
        
    def Models(self, modeltype,input_channels,n_classes,channel_sizes,kernel_size,dropout,
               input_num,hidden_size,num_layers):
        
        if modeltype =='Wavenet':
            model= wavenetmodel.WaveLSTM(in_depth=90, res_channels=8, skip_channels=8, dilation_depth=5,kernel_size=7, n_repeat=1)
            
            
        if modeltype =='TCN':
            model= models.TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
            
            
        if modeltype=='TCN_BLSTM':
#            model= TCN_BiLTM3.TCN_BLSTM(input_num+1,hidden_size, num_layers,input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
            model= tcnblstmmodel.TCN_BLSTM(input_channels, n_classes, channel_sizes, hidden_size, num_layers, kernel_size=kernel_size, dropout=dropout)
            
            
        if modeltype =='BLSTM':            
            model =Blstm.Bi_LSTM(input_num+1, hidden_size, num_layers, n_classes)
            
        if modeltype =='LSTM':            
            model =LSTM.LSTM(input_num+1, hidden_size, num_layers, n_classes)
            
            
        if modeltype =='GRU':
            model=GRU.GRU(input_num+1, hidden_size, num_layers, n_classes)

        return model
 
    
    
    
    
    
    
        
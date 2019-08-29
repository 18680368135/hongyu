#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:11:58 2019

@author: zjr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class WavenetBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, kernel_size):
        super(WavenetBlock, self).__init__()
        
        
        self.batchnormal=nn.BatchNorm1d(8)
    
    
    
        self.diatedconv=nn.Conv1d(in_channels=res_channels, out_channels=res_channels,
                                               dilation=dilation,padding=((kernel_size-1) * dilation)/2,
                                               kernel_size=kernel_size)
        
        
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, 
                                               kernel_size=1,stride=1,padding=0,groups=1, bias=False)         
        
        
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)
    
        

    def init_weights(self):
        
        self.diatedconv.weight.data.normal_(0, 0.01)
        self.skip_conv.weight.data.normal_(0, 0.01)
        self.residual_conv.weight.data.normal_(0, 0.01)


    def forward(self, inputs):
        
        batch_nor=self.batchnormal(inputs)

        diateout=self.diatedconv(batch_nor)
            
        #---------------------------------       
        skip_out = self.skip_conv(diateout) 
              
        #---------------------------------        
        res_out = self.residual_conv(diateout)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        
  
        
        return res_out , skip_out






class SeriesnetNet(nn.Module):
    def __init__(self, in_depth, res_channels, skip_channels, dilation_depth,kernel_size,n_repeat):
        super(SeriesnetNet, self).__init__()
        
        self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        #diated conv1d  return 2 parameters
        
        self.main = nn.ModuleList([WavenetBlock(res_channels,skip_channels,dilation,kernel_size) for dilation in self.dilations])
        
     
        self.pre_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels,
                                              kernel_size=31, stride=1,padding=1, dilation=1, groups=1, bias=False)
    
    
      
        self.relu=nn.ReLU()
        self.conv=nn.Conv1d(in_channels=skip_channels,out_channels=1,kernel_size=1)
    
    

        self.lstm = nn.LSTM(input_size=8, hidden_size=20, num_layers=1, batch_first=True, bidirectional=False)
      
        self.lstmfc=nn.Linear(20,1) 
        
        self.init_weights()
    
    
 
    def init_weights(self): 
        self.pre_conv.weight.data.normal_(0, 0.01)
        self.conv.weight.data.normal_(0, 0.01)
        self.lstmfc.weight.data.normal_(0, 0.01)
        
        

    def forward(self,inputs):        
        #causal conv       
        convoutputs = self.preprocess(inputs)
#        ----------------------------Conv1---------------------------------
        skip_connections = []     
        
        for layer in self.main:
            outputs,skip = layer(convoutputs)
            skip_connections.append(skip)        
            
        outputs = sum([s[:,:,-outputs.size(2):] for s in skip_connections])  
        
        outputs = self.relu(outputs) 
        outputs=self.conv(outputs)
#        ----------------------------LSTM----------------------------------
        
        
        lstminputs=convoutputs.permute(0,2,1)  
        
        h0 = torch.zeros(1, lstminputs.size(0), 20)# 2 for bidirection 
        c0 = torch.zeros(1, lstminputs.size(0), 20)     
        
        lstmcell,_=self.lstm(lstminputs,(h0,c0))
        lstmoutputs=self.lstmfc(lstmcell)#(batch_szie,90,1)
        lstmoutputs=lstmoutputs.permute(0,2,1)

#        ----------------------------LSTM+Wavenet----------------------------------
      
        
        output = outputs * lstmoutputs 
 
        output=output.permute(0,2,1)
 
        return output
    
    
    
    def preprocess(self,inputs):  
        out = self.pre_conv(inputs)     
        return out
    
    
    
    
    

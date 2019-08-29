#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:17:38 2019

@author: zjr
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=30, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
               
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
    
    def forward(self, inputs):
     
#        print "This is the original inputs' shape :",inputs.shape
        outputs = super(CausalConv1d, self).forward(inputs)     
        return outputs[:,:,:-1]



class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, dilation, padding=6, kernel_size=7, stride=1,
                ):       
        super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation )
   
    def forward(self, inputs):
        
        outputs = super(DilatedConv1d, self).forward(inputs)
        
        return outputs



class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, kernel_size):
        super(ResidualBlock, self).__init__()
        

        self.filter_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation, padding=((kernel_size-1) * dilation)/2, kernel_size=kernel_size)
        self.gate_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation,padding=((kernel_size-1) * dilation)/2,kernel_size=kernel_size)      
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, kernel_size=1,stride=1,padding=0,groups=1, bias=False)      
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)
        
    def forward(self,inputs):
        
        #---------------------------------       


        sigmoid_out = F.sigmoid(self.gate_conv(inputs))       
        tahn_out = F.tanh(self.filter_conv(inputs))    
        output = sigmoid_out * tahn_out  
    
        #---------------------------------       
        skip_out = self.skip_conv(output)    
        #---------------------------------        
        res_out = self.residual_conv(output)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        
        
        return res_out , skip_out





#res_channels=inputchannels and output channels
#dialation_depth should be like length([12 ,12 ,12 ,12 ,12 ,12 ])
class WaveNet(nn.Module):
    def __init__(self, in_depth, res_channels, skip_channels, dilation_depth,kernel_size,n_repeat):
       
        super(WaveNet, self).__init__()
        
        self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        #diated conv1d  return 2 parameters
        
        self.main = nn.ModuleList([ResidualBlock(res_channels,skip_channels,dilation,kernel_size) for dilation in self.dilations])
        

        
        self.pre_conv = CausalConv1d(in_channels=res_channels, out_channels=skip_channels)
    
   

        self.post = nn.Sequential(nn.ReLU(),
                                  nn.Conv1d(in_channels=skip_channels,out_channels=1,kernel_size=1) )  
    
    
        
        
        self.last=nn.Sequential(nn.ReLU(),
                                nn.Conv1d(skip_channels,skip_channels,1),
                                nn.ReLU(),
                                nn.Conv1d(skip_channels,in_depth,1))
                               

    
    
    def forward(self,inputs):
        
        #causal conv
        
        convoutputs = self.preprocess(inputs)
#        ----------------------------Conv1---------------------------------
        skip_connections = []     
        
        for layer in self.main:
            outputs,skip = layer(convoutputs)
            skip_connections.append(skip)        
            
        outputs = sum([s[:,:,-outputs.size(2):] for s in skip_connections])     
        outputs = self.post(outputs) 

#        ----------------------------LSTM----------------------------------
        
 
        return outputs
    
    
    
    def preprocess(self,inputs):  
        out = self.pre_conv(inputs)     
        return out
    
    
    
    

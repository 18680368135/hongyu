#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:50:59 2019

@author: zjr
"""

from torch import nn
import Wavenet2



class WaveLSTM(nn.Module):
    def __init__(self,  in_depth, res_channels, skip_channels, dilation_depth,kernel_size, n_repeat):
        super(WaveLSTM, self).__init__()  
        
        self.Wavenet = Wavenet2.SeriesnetNet(in_depth=in_depth, res_channels=res_channels, skip_channels=skip_channels, dilation_depth=dilation_depth,kernel_size=kernel_size, n_repeat=n_repeat)        
        self.linear = nn.Linear(in_depth, n_repeat)#[90 1]      
        self.relu=nn.ReLU()
        self.init_weights()
        
        
        
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 =  self.Wavenet(x)      
        y1=   self.relu(y1)
        y1 =  self.linear(y1[:, :, -1])
        
        
        return y1
    
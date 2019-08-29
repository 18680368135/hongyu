#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:24:42 2019

@author: zjr
"""

from torch import nn
import TCN_BLSTM3


class TCN_BLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_channels, hidden_size, num_layers, kernel_size, dropout):
        super(TCN_BLSTM, self).__init__()  
        self.tcnblstm = TCN_BLSTM3.TemporalConvNet(input_size, num_channels,hidden_size,num_layers, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcnblstm(x)
        return self.linear(y1[:, :, -1])
    
    
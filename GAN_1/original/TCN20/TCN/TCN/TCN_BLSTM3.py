#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:23:48 2019

@author: zjr
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        
        
        
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
     
        
        self.chomp1 = Chomp1d(padding)

        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)



        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)


        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        
        self.relu = nn.ReLU()
        self.init_weights()
        
        

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)



    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)






class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,hidden_size,num_layers,kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.num_layers=num_layers
        self.hidden_size=hidden_size
    
        
        
        for i in range(num_levels):
            
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
#            print "inchannel",in_channels
            out_channels = num_channels[i]
#            print "outchannel",out_channels
            
            if dilation_size == 1:
                self.lstmlayer=nn.LSTM(8,hidden_size, num_layers, batch_first=True, bidirectional=False)
                
            else:
#                print dilation_size
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
      
 
        self.network = nn.Sequential(*layers)
    
    
    
    
    def forward(self, x):
        
        
        x=x.permute(0,2,1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()# 2 for bidirection 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()    
        out, _ = self.lstmlayer(x,(h0,c0))  # #[batch_size,5,40]  hiddensize*2    
     
        out=out.permute(0,2,1)
        
     
        y=self.network(out)
        
#        print self.network

        return y
    
    
    
    
    
    
    
    

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:35:05 2019

@author: zjr
"""
from torch import nn
import torch
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.fullyunit1=150
        self.fullyunit2=12
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)  # 2 for bidirection

        
    def forward(self, x):

        print(x.shape)
#-------------------------------------------------
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda() # 2 for bidirection 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
#-------------------------------------------------        

 #       h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # 2 for bidirection 
 #       c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)


        out = self.fc(out[:, -1, :])

#        print out.shape
        
        
        return out
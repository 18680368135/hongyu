from torch import nn
import torch







class Bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Bi_LSTM, self).__init__()
        self.fullyunit1=150
        self.fullyunit2=12
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
#-------------------------------------------------
#        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda() # 2 for bidirection 
#        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
#-------------------------------------------------        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
#        print out.shape
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])


        
        return out
    
    
    
    
#class Bi_LSTM(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers, num_classes):
#        super(Bi_LSTM, self).__init__()
#        self.input_size=input_size
#        self.hidden_size = hidden_size
#        self.num_layers = num_layers
#        self.num_classes = num_classes
# 
#        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)           
#        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
#
#    def forward(self, x):
#        # Set initial states
#        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda() # 2 for bidirection 
#        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
#        # Forward propagate LSTM
#        x=x.permute(0,2,1)
#            
#        out, _ = self.lstm(x,(h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
#        # Decode the hidden state of the last time step
#      
#    
#        return self.fc(out[:, -1,: ])

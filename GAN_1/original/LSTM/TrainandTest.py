#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:19:33 2019

@author: zjr
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import Movingdata_Pytorch_Original
import warnings

warnings.filterwarnings("ignore")  



class TrainandTest:
    
    def __init__(self, epoch,model,original_data,batch_size,lr,input_num,test_day,train_start ,test_end,train_end,cuda,optims,clip,log_interval):
        
        self.epoch=epoch
        self.original_data=original_data
        self.lr=lr
        self.input_num=input_num
        self.test_day=test_day
        self.train_start=train_start
        self.test_end=test_end
        self.train_end=train_end
        self.cuda=cuda
        self.optims=optims
        self.model=model
        self.batch_size=batch_size
        self.clip=clip
        self.log_interval=log_interval
        
      
    def getdataset(self,original_data,input_num,test_day,start ,end, middle,train):
        X_train,X_test,Y_train,Y_test,ss_y=Movingdata_Pytorch_Original.get_all_data(original_data,input_num,test_day,start ,end, middle,train)  
        return X_train,X_test,Y_train,Y_test,ss_y  
  
    
    
    def returndata(self,inputs,Y_test,ss_y):
        predictlabel=ss_y.inverse_transform(inputs)
        Y_test=Y_test.detach().cpu().numpy()
        reallabel=ss_y.inverse_transform(Y_test)
        return predictlabel,reallabel
    
    
    
    def train_evaluate(self,epoch,model,original_data,batch_size,lr,input_num,test_day,train_start,test_end,train_end,cuda,optims,clip,log_interval):          

        # print(original_data.shape)

        X_train,X_test,Y_train,Y_test,ss_y=self.getdataset(original_data,
                                  input_num,test_day,train_start ,test_end, train_end,train=True)       
        optimizer= getattr(optim, optims)(model.parameters(), lr=lr)
       
        
        for ep in range(1, epoch+1): 
      
            model.train()
            batch_idx = 1
            total_loss = 0            
#-------------------------------------------    
            
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            X_test=X_test.cuda()
            Y_test=Y_test.cuda()
#-------------------------------------------
            for i in range(0, X_train.size()[0], batch_size):
                if i + batch_size > X_train.size()[0]:
                    x, y = X_train[i:], Y_train[i:]
                else:
                    x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
                   
                    
                optimizer.zero_grad()
                    
                output= model(x)
                
                loss = F.mse_loss(output, y)
                
                loss.backward()
     
                if clip > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            
                optimizer.step()
                batch_idx += 1
                total_loss+=loss.item()  
                
                if batch_idx % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    processed = min(i+batch_size, X_train.size()[0])
                    print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.7f}\tLoss: {:.8f}'.format(
                    ep, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))
                    total_loss = 0
                    
           
        model.eval()
        output= model(X_test)   
        predict_output=output.detach().cpu().numpy()
        predict,reallabel=self.returndata(predict_output,Y_test,ss_y)   
        test_loss = F.mse_loss(output, Y_test)
        print("Predict value is:",predict)
        print("Real value is:",reallabel)
        print('Test set: Average loss: {:.6f}\n'.format(test_loss.item()))
        
        return predict,reallabel
    
    
   
  
    def validation(self,epoch,model,original_data,batch_size,lr,input_num,test_day,train_start,test_end,train_end,cuda,optims,clip,log_interval):     
        validation_size=10
       
        X_train,X_test,Y_train,Y_test,ss_y=self.getdataset(original_data,                                  
                                  input_num,test_day,train_start ,test_end, train_end,train=True) 
        
        optimizer= getattr(optim, optims)(model.parameters(), lr=lr)
        current_loss=[]
        validation_loss=[]
        for ep in range(1, epoch+1): 
      
            model.train()
            batch_idx = 1
            total_loss = 0
            total_loss_v=0    
            
#-------------------------------------------    
#            if cuda:
#                X_train = X_train.cuda()
#                Y_train = Y_train.cuda()
#                X_test=X_test.cuda()
#                Y_test=Y_test.cuda()
#-------------------------------------------
            for i in range(0, X_train.size()[0], batch_size):
                if i + batch_size > X_train.size()[0]:
                    x, y = X_train[i:i+1], Y_train[i:i+1]
                    x_v,y_v=X_train[i+1:],Y_train[i+1:]
                else:
                    x, y = X_train[i:(i+batch_size-validation_size)], Y_train[i:(i+batch_size-validation_size)]  
                    x_v,y_v=X_train[(i+batch_size-validation_size):(i+batch_size)],Y_train[(i+batch_size-validation_size):(i+batch_size)]
                   
                    
                optimizer.zero_grad()
                    
                output= model(x)
                
                loss = F.mse_loss(output, y)
                
                loss.backward()
     
                if clip > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            
                optimizer.step()
                batch_idx += 1
                total_loss+=loss.item()  
                
                
                model.eval()
                
                output_v= model(x_v)
                loss_v= F.mse_loss(output_v, y_v)
                total_loss_v+=loss_v.item()
     
                if batch_idx % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    cur_loss_v= total_loss_v /log_interval
                    
                    current_loss.append(cur_loss)
                    validation_loss.append(cur_loss_v)
                    processed = min(i+batch_size, X_train.size()[0])
                    print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                            ep, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))

                    total_loss = 0
                    total_loss_v=0
                    
        return current_loss,validation_loss
        
        
        
        
        
        
        
        
        
        






































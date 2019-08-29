# -*- coding: utf-8 -*-
###########################################
# CPU VERSION 1.0(updated on June 11,2018)
#    调用64个CPU核心进行运算
#v7采用train阶段的mean和std对test和label数据进行变换
#这个版本（V8）在V7基础上改进架构，增强可读性和可修改行
###########################################
'''
1-9-1

和标签之间的误差。
（2）输入维度
输入只有9个维度，即股票的pctchange,成交量，成交价，最高价、最低价、开盘价、收盘价，指pctchange，指数成交量。
（3）1层LSTM;40个隐含层,所有输入都是同一只待测股票的重复 batch_size=60 forget=1.0
（4）offset=0
（5）训练数据转换成当日和前一日之间涨幅比例代替，去除数据绝对值不同的影响。即 （当日数据-前一日数据）/前一日数据
（7）训练数据再进行归一化到(0,1)区间，采用以下公式：当日数据=(当日数据-最小数据)/(最大数据-最小数据)，其中最大数据
是训练集最大数据，最小数据是训练集最小数据
（8）第1和第8维度由（0,1）之间的随机数代替，且每轮训练数据，随机数不一样。
（9）除待预测维度（最高价或者最低价），其他维度数据在每轮训练中都乘以一个和训练次数相关的线性缩小因子，即(time-i-1)/time
time是训练次数，i是当前训练次数。训练完成时，只有带预测维度的输入数据有值，其他维度数据都为零。
（10）首次随机生成的权重集合，或者前一日期训练结果的权重，用来当前日期训练，训练结果有可能不准确。因此需要重复训练，取第一次
训练以后的训练过权重来预测
'''
import copy
import pymysql
import pandas as pd
import numpy as np
import os
import gc
import random
import time
#import requests
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
from sqlalchemy import create_engine
import getInputData_v6 as getInputData
import datetime


#============== CPU 版本 ===============================
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 禁用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#=======================================================

def get_train_data(batch_size, time_step, train_begin, train_end, data_):
    batch_index = []  # 定义索引
    '''
    data_train train_begin到train_end条 多维数据 
    normalized_train_data 标准化后的多维数据
    '''
    #截取数据，获得训练数据
    data_train = data_[train_begin:train_end]  # 训练数据

    #训练集两行数据异常，去掉异常，用相邻数据补充。
    #data_train[1527,:]=data_train[1526,:] 
    #data_train[1528,0]=data_train[1529,0]
    #data_train[1523,1:3]=data_train[1522,1:3]
    #data_train[1096,1:3]=data_train[1095,1:3]
    #data_train[727,1:3]=data_train[726,1:3]
    #data_train[885,1:3]=data_train[884,1:3]
    #data_train[440,1:3]=data_train[439,1:3]
    #(pd.DataFrame(data_train)).to_csv('data.csv', mode='a')

    #定义存放每一维数据最大值和最小值的数组
    data_max=[0 for i in range(data_train.shape[1])]
    data_min=[0 for i in range(data_train.shape[1])]

    #获取每一维数据的最大值和最小值
    for i in range(data_train.shape[1]):
        data_max[i]=max(data_train[:,i])
        data_min[i]=min(data_train[:,i])

    #最高价和最低价两列数据的最大值和最小值分别设为0.1和-0.1
    #少数数据值存在异常，顾导致获得异常的最大值或者最小值。为了屏蔽该异常，认为设定合适的
    #最大值和最小值。对于少数异常数据，相当于噪声，应该不影响结果。
    #data_max[5],data_max[6],data_max[9]=0.1,0.1,0.1
    data_max[1],data_max[2]=1.0,1.0
    #data_min[5],data_min[6],data_min[9]=-0.1,-0.1,-0.1

    #数据矩阵化
    data_max=np.array(data_max)
    data_min=np.array(data_min)
    #print('data_max=',data_max,'   data_min=',data_min)

    #数据归一化到(0,1)区间
    normalized_train_data=(data_train-data_min)/(data_max-data_min)
    #(pd.DataFrame(data_train)).to_csv('data_train.csv', mode='a')

    #(pd.DataFrame(normalized_train_data)).to_csv('normalized_data.csv', mode='a')

    #数据按照batch进行分段，形成最终训练集
    train_x, train_y = [], []  # 训练集

    #i最大值为(len(normalized_train_data)-time_step+1)-1
    #i是用作区间表示，故数组下标右区间最大值可以是len(normalized_train_data),再加上循环
    #最大值+1=range（）中的上限。
    for i in range(len(normalized_train_data) - time_step+1):
        # 如果i为 一个batch的起点，加入此索引
        if i % batch_size == 0:
            batch_index.append(i)
        # 将多维输入数据分堆
        x = normalized_train_data[i:i + time_step, :input_size]
        # 将标签分堆
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    # 加入最后一个的batch索引，该batch个数不定，即筹不够60日的尾数
    if i % batch_size != 0:
        batch_index.append(len(normalized_train_data)-time_step)
    
    #(pd.DataFrame(train_x[-1])).to_csv('train_x.csv', mode='a')
    #(pd.DataFrame(train_y[-1])).to_csv('train_y.csv', mode='a')

    return batch_index, train_x, train_y, data_max, data_min


def get_test_data(time_step, test_begin, test_end, data_,data_max,data_min):

    '''
    data_train train_begin到train_end条 多维数据 
    normalized_train_data 标准化后的多维数据
    '''
    data_test = data_[test_begin:test_end]  # 训练数据
    normalized_test_data=(data_test-data_min)/(data_max-data_min)
    #(pd.DataFrame(normalized_test_data)).to_csv('normalized_data.csv', mode='a')

    #=====================
    #将非带预测维度的数据统一设置为零
    #column_number=normalized_test_data.shape[1]                
    #for i in range(column_number):
     #   if (i%7)==0:    
      #      normalized_test_data[:,i]=0

    '''
    if ifhigh==1:
        for i in range (5): #列i设置有问题
            normalized_test_data[:,i]=0
        for i in range(7,column_number-1):
            normalized_test_data[:,i]=0
    elif ifhigh==0:
        for i in range (5):
            normalized_test_data[:,i]=0
        for i in range(7,column_number-1):
            normalized_test_data[:,i]=0
    '''        


    test_x, test_y = [], []  # 训练集
    # 从0到数据大小 - 步数
    for i in range(len(normalized_test_data) - time_step+1):
        # 如果i为 一个batch的起点，加入此索引
        # 将多维输入数据分堆
        x = normalized_test_data[i:i + time_step, :input_size]
        # 将标签分堆
        y = normalized_test_data[i:i + time_step, input_size, np.newaxis]
        test_x.append(x.tolist())
        test_y.append(y.tolist())
    #(pd.DataFrame(test_x[-1])).to_csv('test_x.csv', mode='a')
    #(pd.DataFrame(test_y[-1])).to_csv('test_y.csv', mode='a')

    return  test_x, test_y


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]

    w_in = weights['in']
    b_in = biases['in']

    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入

    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入

    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)
        
    #[cell]*2为2层LSTM
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, input_rnn, initial_state=init_state,dtype=tf.float32)
    #output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])

    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states

# ————————————————训练模型————————————————————
'''
batch_size batch大小
time_step 训练步数
train_begin 训练开始位置
train_end 训练结束位置
'''
def train_lstm(which, train_x, train_y, batch_index, time_step, ifhigh):
    with tf.variable_scope("sec_lstm_%s_%d_%d" % (stock_code, ifhigh, which), reuse=tf.AUTO_REUSE):

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
        L=tf.placeholder(tf.float32)
        pred, _ = lstm(X)
        pred_2 = tf.reshape(pred, [-1, time_step])
        loss = tf.reduce_mean(tf.abs(tf.reshape(pred_2[:, -1], [-1]) - tf.reshape(Y[:, -1], [-1])))
        # loss = tf.reduce_mean(tf.abs(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        #new_lr=lr            
        # 优化器定义
        train_op = tf.train.AdamOptimizer(L).minimize(loss)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        with tf.Session() as sess:
            
            if os.path.isfile('./model_save_%s_%d_%d/checkpoint' % (stock_code, ifhigh, which)):
                module_file = tf.train.latest_checkpoint('./model_save_%s_%d_%d/' % (stock_code, ifhigh, which))
                saver.restore(sess, module_file)
                print('reload model')
            else:
            
                print('setup new model')
                sess.run(tf.global_variables_initializer())

            #batch_index, train_x, train_y, mean, std = get_train_data(batch_size, time_step, train_begin, train_end, data_,1,ifhigh)
            len_batch=len(batch_index)
            for i in range(iteration_num):
                loss_temp_1=0
                loss_temp_2=0.000001
                loss_max=loss_temp_2
                '''
                if i<first_num:
                    new_lr=lr
                else:
                    j=i-first_num
                    second_num=iteration_num-first_num
                '''                    
                new_lr=lr#(lr/2)*(1+(second_num-1-j)/second_num)
                for step in range(len_batch-2,-1,-1):
                    #if random.uniform(0,1)<tanh((step+2)/len_batch+loss_temp_2/loss_max):
                    if random.uniform(0,1)<((step+2)/(2*len_batch)+loss_temp_2/(2*loss_max)):
                        _, loss_ = sess.run([train_op, loss],
                                        feed_dict={L:new_lr,X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                        loss_temp_1=loss_temp_2
                        loss_temp_2=loss_
                        if loss_>loss_max:
                            loss_max=loss_
                mean_loss=(loss_temp_1+loss_temp_2)/2
                if i%40==0:
                    print('model_%s_%d_%d and i=' % (stock_code, ifhigh, which), i, " loss_max=", loss_max,'  lr=',new_lr)                                
            saver.save(sess,os.path.join(os.getcwd(), './model_save_%s_%d_%d/modle.ckpt' % (stock_code, ifhigh, which)))

    return mean_loss

def tanh(x):
    y=(1.0-np.exp(-2*x))/(1.0+np.exp(-2*x))
    return y


# ————————————————预测模型————————————————————
def prediction(which, test_x, test_y,ifhigh):
    with tf.variable_scope("sec_lstm_%s_%d_%d" % (stock_code, ifhigh, which), reuse=tf.AUTO_REUSE):
        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        #test_x, test_y= get_test_data(time_step, predict_begin, predict_end, data_,data_max,data_min,ifhigh)
        pred, _ = lstm(X)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint('./model_save_%s_%d_%d/' % (stock_code, ifhigh, which))
            saver.restore(sess, module_file)

            #for step in range(len(test_x)):
            predict = sess.run(pred, feed_dict={X: [test_x[-1]]})
            predict=predict.reshape(-1)
            

    return predict[-1]

#将股票数据转化为当前与昨天的差值的比例作为输入数据。，
def data_difference(data,step_number):
    raw_number=data.shape[0]
    column_number=data.shape[1]
    data_trans=copy.deepcopy(data)
    for i in range (column_number):
        if (i%7)!=0:
            for j in range (raw_number):
                if j<step_number:
                    data_trans[j,i]=0
                else:
                    data_trans[j,i]=(data[j,i]-data[j-step_number,i])/data[j-step_number,i]

    data_label=data_trans[:,-1]
    return data_trans, data_label

def run(offset, time_step,stock_code_, name, code_num):
    #用于记录最高价（最低价）的值。
    test_label_high=[]
    test_label_low=[]
    predict_high=[]
    predict_low=[]
    losses = []
    global today_judge
    global today
    global stock_code
    global batch_size 

    #stock_code_是code_raw，即股票代码列，code_num则是程序所用到股票代码的数组。
    stock_code = stock_code_[code_num[0]]
    code_num_len=len(code_num)
    for j in range (code_num_len):
        if j==0:
            data_origin_total,index_origin = getInputData.get_data_origin(code=stock_code_[code_num[j]], debug=False)
        else:
            data_origin,_ = getInputData.get_data_origin(code=stock_code_[code_num[j]], debug=False)
            data_origin_total=pd.merge(data_origin_total,data_origin,on='date')
    
    # 第一个参数为label编号，即label天后的数据；第二个参数为股票代号
    predict_=0
    for j in range (5):#
        data = getInputData.get_data(label_num=j+1, df1=data_origin_total,df2=index_origin, is_high=True, contains_index=True, debug=False)
        data_2=copy.deepcopy(data)
        #(pd.DataFrame(data_2)).to_csv('data_2.csv', mode='a')
        data = data.iloc[:, 1: 1 + input_size + output_size].values  # 取第2-11列
        #(pd.DataFrame(data)).to_csv('data.csv', mode='a')
        data, data_label_high=data_difference(data,j+1)
        p_start = len(data) - 1 - offset-j  #
        p_end = len(data) - offset

        #label_high=data[p_start-1,-1]
        #print(label_high,'   ',data[p_start,-1])
        label_high=data_2['label'][p_end-1]
        today = data_2['date'][p_end-1]        #print(label_high)
            #print('today=','\n',today)


        #参数P_start+50,即将测试集放入训练集中训练。测试的结果是已经训练过的结果。
        #用于考察训练的loss是否足够小。
        #如果将50删除，即训练集和测试集不重叠。属于正常的测试。
        batch_index, train_x, train_y, data_max, data_min = get_train_data(batch_size, time_step, j+1, p_start, data)
        loss = train_lstm(j+1, train_x, train_y, batch_index, time_step, 1)

        losses.append(loss)

    #p_start-100,即预测从p_start前100个开始，但只取最后一个。因为lstm会记忆历史输入，
    #在测试时，多测试前100个历史数据，在测试最后一个输入数据时，就有比较准确的历史记忆信息。

        test_x, test_y= get_test_data(time_step,0, p_end, data,data_max,data_min)
        predict_ = prediction(j+1, test_x,test_y,1) #0 replace p_start-100

        predict_ = (data_max[-1]-data_min[-1])*predict_+data_min[-1]

        #print('predict=',predict_)
        #data_3=np.vstack((data_3,data_predict))
        #data_3[p_end,5]=predict_


        predict_=(predict_+1)*label_high
        if offset-j<=0:
            label_high_true=predict_
        else:
            label_high_true=data_2['label'][p_end-1]
            
        predict_high.append(predict_.tolist())

        test_label_high.append(label_high_true)
        
    del data
    del data_2
    gc.collect()
    
    predict_low = []

        # 第一个参数为label编号，即label天后的数据；第二个参数为股票代号
    predict_=0
    for j in range (5):  #j=连续预测的天数  
        data = getInputData.get_data(label_num=j+1, df1=data_origin_total,df2=index_origin, is_high=False, contains_index=True, debug=False)
        
        data_2=copy.deepcopy(data)

        data = data.iloc[:, 1: 1 + input_size + output_size].values  # 取第2-8列
        data, data_label_low=data_difference(data,j+1)
        p_start = len(data) - 1 - offset-j
        p_end = len(data) - offset


        #参数P_start+50,即将测试集放入训练集中训练。测试的结果是已经训练过的结果。
        #用于考察训练的loss是否足够小。
        #如果将50删除，即训练集和测试集不重叠。属于正常的测试。
        #print('loss=',loss)
        #p_start-100,即预测从p_start前100个开始，但只取最后一个。因为lstm会记忆历史输入，
        #在测试时，多测试前100个历史数据，在测试最后一个输入数据时，就有比较准确的历史记忆信息。
        #if j==0:
        label_low=data_2['label'][p_end-1]
            #label_low=predict_ #将前一步预测值作为label_low，用于从相对值恢复出绝对值 423行
        batch_index, train_x, train_y, data_max, data_min = get_train_data(batch_size, time_step, j+1, p_start, data)
        loss = train_lstm(j+1, train_x, train_y, batch_index, time_step, 0)
        
        losses.append(loss)
        test_x, test_y= get_test_data(time_step,0, p_end, data,data_max,data_min)
        predict_ = prediction(j+1, test_x,test_y,0) #0 replace p_start-100
        predict_ = (data_max[-1]-data_min[-1])*predict_+data_min[-1]

        #print('predict=',predict_)





        predict_=(predict_+1)*label_low #用于恢复出绝对值
        if offset-j<=0:# 当offset-j=0时，有数据最后一天。offset-j<0，意味着预测未来的数据
            label_low_true=predict_#data_2['label'][len(data_2)-1] #统一用已经发生的最后一天的标签表示，是不正确的数据，基金是填充需要
        else:
            label_low_true=data_2['label'][p_end-1] #用已经发生的预测那天的标签表示。
        predict_low.append(predict_.tolist())
        test_label_low.append(label_low_true)
        #print('j=',j,'predict=',predict_,'label=',label_low_true)

    del data
    del data_2
    gc.collect()
    
    predict=np.dstack((predict_high,predict_low))
    predict=predict.reshape(-1,2)
    print('predict=')
    print(predict)
    
    test_label=np.dstack((test_label_high,test_label_low))
    test_label=test_label.reshape(-1,2)
    print('test_label=')
    print(test_label)

    loss = np.mean(losses) * 100 #5天测试的平均误差
    
    loss = float(loss)
    np.savez('./1-9-9.npz',predict, test_label,loss,np.array(today),lr,stock_code)
    print('today=',today,'   stock_code=',stock_code)
    if today_judge!=today and today_judge!='':
        data_change=True
    else:
        data_change=False
    
    return predict, test_label,data_change


if __name__ == "__main__":
    pymysql.install_as_MySQLdb()

    rnn_unit = 20  # 隐层数量
    time_step=20#输入神经元个数
    #label_num = 1 #预测未来天数
    #input_size = 37 #输入数据（每个神经元输入数据）的维数
    output_size = 1 #输出数据的维数
    batch_size = 60 #每批训练的输入向量个数

    #my_config = tf.ConfigProto()
    #my_config.log_device_placement = False  # 输出设备和tensor详细信息
    #my_config.gpu_options.allow_growth = True  # 基于运行需求分配显存(自动增长)

    # 输入层、输出层权重、偏置
    '''
    rnn_unit 隐层数
    weights 权重 输入多维 输出1维
    biases  偏差
    '''
    #k=1610  #股票东方财富300059在数据库中的序号1610,中国平安988,同花顺1899恒瑞医药1345
    #offset=0#92 #最近股票数据后推offset天，作为训练数据集。offset天之后的5天，作为测试数据集
    #time = 1 #训练轮数
    #lr = 0.003  # 学习率
    lr_group=[[0.0006,50]]
    code_num=[1610]  #300059最新代号是3600
    input_size=len(code_num)*7+2
    codes_raw, names_raw = getInputData.get_code() 
    '''
    i=0
    while codes_raw[i]!='300059':
        i=i+1
    print(i)
    '''

    offset_start=120 #对offset进行赋起始值
    simu_num=1 #独立的模拟次数，用于结果值取平均值
    #=========================
    '''
    flag=True
    while flag:
        nowtime=datetime.datetime.now().strftime('%H')#现在 
        if nowtime>2 and nowtime<4:
            Flag=False
            offset_start=0
            simu_num=11
        else:
            time.sleep(1800)
    '''
    #====================        


    lr_group_len=len(lr_group)
    offset_end=-1 #对offset进行赋终止值
    #while True:
        #offset=0
    for offset in range (offset_start,offset_end,-1):
        predict_offset=[]
        test_label_offset=[]

        for i in range (simu_num):
            j=0
            data_change=False
            today_judge=''
            today=''
            for j in range (lr_group_len):
                array_lr_group=np.array(lr_group)
                lr=array_lr_group[j,0]
                iteration_num=int(array_lr_group[j,1])
                first_num=iteration_num/2

#============================================================================
                tf.reset_default_graph() 

                weights = {
                    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
                    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
                    }

                biases = {
                    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
                    'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
                    }
#============================================================================


                # 获取训练集
                predict, test_label,data_change=run(offset, time_step,codes_raw, names_raw,code_num)
            
                predict_offset.append(predict.tolist())
                test_label_offset.append(test_label.tolist())
                loss_predict = (predict-test_label)*100/test_label
                loss_predict_average = np.mean((np.array(predict_offset)-np.array(test_label_offset))*100/np.array(test_label_offset),axis=0)
                print('offset=',offset)
                print('loss_predict=')
                print(loss_predict) 

        predict_offset=np.array(predict_offset)
        test_label_offset=np.array(test_label_offset)
        predict_offset.sort(axis=0)
        loss_predict_average = np.mean(abs(predict_offset[int(i/2),:]-test_label_offset[int(i/2),:])*100/test_label_offset,axis=0)

        #print('offset=',offset,'  final predict= \n',predict_offset[int(i/2),:],'  test_label= \n',test_label_offset[int(i/2),:],'  loss_predict_final= \n',loss_predict_average,)
        np.savez('./1-9-8.npz',predict_offset[int(i/2),:],test_label_offset[int(i/2),:],loss_predict_average,np.array(today))


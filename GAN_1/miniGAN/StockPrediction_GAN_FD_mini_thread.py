﻿# encoding=utf-8
from DataProcessor import DataProcessor
from GANFD_mini import GANFD
import pymysql, os
import numpy as np
import pandas as pd
import multiprocessing
import time



##################################### hyperparameters ########################################
params = dict(
    testDays=50,
    batchSize=32,
    timeStep=21,
    hiddenUnit=32,
    GeneratorInputSize=1,
    GeneratorOutputSize=1,
    discriminatorInputSize=1,
    dim=16,
    c=0.001,
    learningRate=0.00001,
    k=1,
    epochs=150,
    outputGraph=False
)
"""
**params 表示关键字参数
"""
###############################################################################################
gpu_limit = 3
gpu_num = 8
# GPU_useage = [gpu_limit] * gpu_num
GPU_useage = multiprocessing.Array('i', [4, 4, 4, 0, 4, 4, 4, 4])
device = multiprocessing.Array('i', [0, 1, 2, 3, 4, 5, 6, 7])


def get_stable_predict(predict_list):
    # print(predict_list)
    predict_list.remove(min(predict_list))
    predict_list.remove(max(predict_list))
    stable_predict = np.mean(predict_list)
    # print(stable_predict)
    return stable_predict


def get_gpu_device(lock):
    my_gpu = -1
    while True:
        with lock:
            for i in device:
                if GPU_useage[i] > 0:
                    my_gpu = i
                    GPU_useage[i] -= 1
                    break
            if my_gpu != -1:
                return my_gpu
            time.sleep(3)


def release_gpu(device, release_lock):
    with release_lock:
        GPU_useage[device] += 1


def dataprocessandpredict(q, dataProcessor, day, lock, release_lock):
    device = get_gpu_device(lock)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(device)
    realdata, trainDf = dataProcessor.getData(day)
    # normalizatedData, valueMax, valueMin = dataProcessor.normalization(trainDf.iloc[:, :].values)
    smoothTrainData, last_close = dataProcessor.dataSmooth(trainDf)
    normalizatedData, valuemax, valuemin = dataProcessor.normalization(smoothTrainData)
    predict_list = []
    model = GANFD(**params)
    for i in range(5):
        predict = model.trainAndPredict(normalizatedData, dataProcessor, day, os.getppid(), os.getpid())
        predict_list.append(predict[-1, -1, -1])
    stable_predict = get_stable_predict(predict_list)
    predictprice = stable_predict * (valuemax[-1] - valuemin[-1]) + valuemin[-1]

    # predictprice = predict[-1, -1, -1] * (valuemax[-1] - valuemin[-1]) + valuemin[-1]
    predictprice = last_close * (1 + predictprice)
    q.put([day, realdata, predictprice])
    release_gpu(device, release_lock)


def readqueue(q):
    df = pd.DataFrame(columns=["day", "realdata", "predict"])
    while True:
        queuedata = q.get(True)
        df = df.append({'day': queuedata[0],
                        'realdata': queuedata[1],
                        'predict': queuedata[2]}, ignore_index=True)
        if(q.empty()):
            break
    df = df.sort_values(by='day', axis=0, ascending=False)
    df.to_csv('./loss_mini/result.csv', index=False, sep=',')
    real_values = df.iloc[:, 1].values
    predict_values = df.iloc[:, 2].values

    return real_values, predict_values


def compute_evaluation_index(realValues, predictValues):
    real_values = np.array(realValues)
    predict_values = np.array(predictValues)

    MSE = np.mean(np.square(real_values - predict_values))
    MAPE = np.mean(np.abs((real_values - predict_values) / real_values)) * 100
    MAE = np.mean(np.abs(real_values - predict_values))

    print('MSE:%.4f' % MSE)
    print('MAPE:%.4f' % MAPE)
    print('MAE:%.4f' % MAE)


def main(pool, q):
    realValues, predictValues = [], []

    df = pd.read_csv('000002.csv')
    # print(df)
    usedDf = df[[
               # 'open', 'high', 'low', 'volume',  # 'turn', 'volume',
               # 'ma',  # Moving averages
               # 'ema',  # Exponential movement index
               # 'bias',  # Bias
               # 'boll', 'boll_ub', 'boll_lb',  # bolling band
               # 'rsi_6', 'rsi_12',  # Relative Strength index
               # 'pdi', 'mdi', 'dx', 'adx', 'adxr',  # Directional movement index
               # 'macd', 'macds', 'macdh',  # MACD
               # 'kdjk', 'kdjd', 'kdjj',  # Stochastic index
               'close',
               'close'  # 预测指标
               ]]
    # usedDf = df[['high']]
    """
    将需要的列名填进去 数量应该为 GeneratorInputSize + 1
    最后一个列为需要预测的指标 !!!
    """
    # print(usedDf)
    dataProcessor = DataProcessor(**params, df=usedDf, inputSize=params["GeneratorInputSize"])
    lock = multiprocessing.Manager().Lock()
    release_lock = multiprocessing.Manager().Lock()
    for day in range(params['testDays'], 0, -1):
        pool.apply_async(func=dataprocessandpredict, args=(q, dataProcessor, day, lock, release_lock))
    pool.close()
    pool.join()
    realvalues, predictvalues = readqueue(q)
    compute_evaluation_index(realvalues, predictvalues)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1代表使用CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  #  "" 也是代表使用CPU
    q = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(28)
    main(pool, q)


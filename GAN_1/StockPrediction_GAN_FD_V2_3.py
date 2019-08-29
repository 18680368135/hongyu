# encoding=utf-8
from DataProcessor import DataProcessor
from ALEGANFD_v3_3 import GANFD
import pymysql, os
import numpy as np
import pandas as pd

##################################### hyperparameters ########################################
params = dict(
    testDays=10,
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
    epochs=300,
    outputGraph=False
)
"""
**params 表示关键字参数
"""
###############################################################################################


def main():
    realValues, predictValues = [], []

    df = pd.read_csv('000001.csv')
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

    for day in range(params['testDays'], 0, -1):
        # print(day)
        model = GANFD(**params)

        realData, trainDf = dataProcessor.getData(day)
        realValues.append(realData)
        # print(trainDf)
        # dataProcessor.dataSmooth(trainDf)
        #normalizatedData, valueMax, valueMin = dataProcessor.normalization(trainDf.iloc[:, :].values)
        # print(normalizatedData)
        smoothTrainData, last_close = dataProcessor.dataSmooth(trainDf)

        # print(last_close)
        # print(smoothTrainData)
        normalizatedData, valueMax, valueMin = dataProcessor.normalization(smoothTrainData)

        predict, mark = model.trainAndPredict(normalizatedData, dataProcessor, day)
        while mark:
            predict, mark = model.trainAndPredict(normalizatedData, dataProcessor, day)
        # print(predict)
        # print(predict.shape)
        # print(predict[-1.2.1.2, -1.2.1.2, -1.2.1.2])
        print(predict[-1, -1, -1], valueMax[-1], valueMin[-1])

        predictPrice = predict[-1, -1, -1] * (valueMax[-1] - valueMin[-1]) + valueMin[-1]
        print(predictPrice)
        predictPrice = last_close * (1 + predictPrice)
        print(predictPrice)
        predictValues.append(predictPrice)



    dataframe = pd.DataFrame({'real': realValues, 'prediction': predictValues})
    dataframe.to_csv('./loss3_2/result.csv', index=False, sep=',')

    real_values = np.array(realValues)
    predict_values = np.array(predictValues)

    MSE = np.mean(np.square(real_values - predict_values))
    MAPE = np.mean(np.abs((real_values - predict_values) / real_values)) * 100
    MAE = np.mean(np.abs(real_values - predict_values))

    print('MSE:%.4f' % MSE)
    print('MAPE:%.4f' % MAPE)
    print('MAE:%.4f' % MAE)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1代表使用CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  #  "" 也是代表使用CPU
    main()


# encoding=utf-8
from DataAccessor import DataAccessor
from DataProcessor import DataProcessor
from model import GAN
import pymysql, os
import numpy as np
import pandas as pd

##################################### hyperparameters ########################################

name = 'root'
password = 'szU@654321'
db_ip = '210.39.12.25'
db_port = '50002'

TIME_STEP = 21                              # 输入神经元个数
BATCH_SIZE = 32                             # 批次大小
INPUT_SIZE = 1                              # Prediction模型的LSTM输入
OUTPUT_SIZE = 1                             # LSTM输出大小
IMG_DIM = 1                                 # 图片的维度
Z_DIM = 100                                 # 输入的随机变量Z的维度

DIM = 16
C = 0.01

PREDICTION_RNN_UNIT = PREVENTION_RNN_UNIT = 32
PREDICTION_INPUT_SIZE = 1
PREVENTION_INPUT_SIZE = 5

PREDICTION_OUTPUT_SIZE = 3
PREDICTION_FC = 2

PREVENTION_OUTPUT_SIZE = 3
PREVENTION_FC = 2

FC2 = 3

LR = 0.00001                                # 学习率

OUTPUT_GRAPH = True                        # 是否输出Tensorboard
EPOCHS = 1000                              # 训练轮次
DAYS = 100         # for all data 4501  2576
###############################################################################################


def main():
    pymysql.install_as_MySQLdb()

    realValues, predictValues = [], []

    accessor = DataAccessor(name, password, db_ip, db_port)
    df = accessor.getData(0)

    dataProcessor = DataProcessor(batch_size=BATCH_SIZE, time_step=TIME_STEP, input_size=INPUT_SIZE, df=df)

    gan = GAN(timeStep=TIME_STEP, dim=DIM, lr=LR, c=C, epoch=EPOCHS,
              predictionInputSize=PREDICTION_INPUT_SIZE, predictionOutputSize=PREDICTION_OUTPUT_SIZE,
              predictionRnnUnit=PREDICTION_RNN_UNIT, predictionFc=PREDICTION_FC,
              preventionInputSize=PREVENTION_INPUT_SIZE, preventionOutputSize=PREVENTION_OUTPUT_SIZE,
              preventionRnnUnit=PREVENTION_RNN_UNIT, preventionFc=PREVENTION_FC,
              fc2=FC2, outputGraph=OUTPUT_GRAPH)

    for day in range(DAYS, 0, -1):
        print(day)

        realData, trainDf = dataProcessor.getData(day)
        realValues.append(realData)

        normalizatedData, valueMax, valueMin = dataProcessor.normalization(trainDf.iloc[:, 1:].values)

        batchIndex, trainX, _ = dataProcessor.getTrainData(normalizatedData)

        if day == DAYS:
            gan.train(batchIndex, trainX)

        predictValue = gan.predict(batchIndex, trainX)

        predictPrice = predictValue * (valueMax[0] - valueMin[0]) + valueMin[0]

        predictValues.append(predictPrice[-1][0])

    dataframe = pd.DataFrame({'real': realValues, 'prediction': predictValues})
    dataframe.to_csv('50daysresult.csv', index=False, sep=',')

    real_values = np.array(realValues)
    predict_values = np.array(predictValues)

    MSE = np.mean(np.square(real_values - predict_values))
    MAPE = np.mean(np.abs((real_values - predict_values) / real_values)) * 100
    MAE = np.mean(np.abs(real_values - predict_values))

    print('MSE:%.4f' % MSE)
    print('MAPE:%.4f' % MAPE)
    print('MAE:%.4f' % MAE)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()


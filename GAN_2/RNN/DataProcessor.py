import numpy as np
import pymysql
import pandas as pd


class DataProcessor(object):
    def __init__(self, batchSize, timeStep, inputSize, df, **kwargs):

        self.batch_size = batchSize
        self.time_step = timeStep
        self.input_size = inputSize

        self.df = df

    def normalization(self, data):
        """
        Effect:
            将数据正则化
        Parameters：
            data: <numpy.ndarray> 需要处理的数据
        Return:
            normalizated_data: <numpy.ndarray> 正则化处理过后的数据
            value_max：<numpy.ndarray> 最大值
            value_min：<numpy.ndarray> 最小值
        """
        value_max = np.max(data, axis=0)  # 对data列方向上求最值
        value_min = np.min(data, axis=0)

        normalizated_data = (data - value_min) / (value_max - value_min)

        normalizated_data = np.nan_to_num(normalizated_data)  # 使用0代替数组normalizated_data中的nan元素，使用有限的数字代替inf元素

        return normalizated_data, value_max, value_min

    def getData(self, days):
        """
        Effect:
            获取模型需要的相关的数据。
        Param：
            days： 天数
            dataType： 数据的类型
            code： 股票代码
        Return:
            test_data: 测试数据
            df_merge： 包含了标签的训练数据集的所有数据
        """
        df = self.df

        test_data = df.iloc[-days, :].values[-1]
        # print("测试数据是：{0}".format(test_data))
        df = df.iloc[:-days, :]
        # print("df : {0}".format(df))

        return test_data, df

    def dataSmooth(self, trainDF):
        trainData = trainDF.iloc[:, :].values
        smoothTrainData = []
        last_close = trainData[-1, -1]
        for i in range(1, len(trainData)):
            smoothTrainData.append((trainData[i] - trainData[i-1]) / trainData[i-1])
        return smoothTrainData, last_close

    def getTrainData(self, normalizated_data):
        """
            处理训练数据
        :param
            normalizated_data: <np.ndarray> 需要处理的数据
        :return:
             batch_index: <list> 批次索引
             train_x: <list> 输入数据（两个用途，1是做Prediction的输入，2是做Discriminator的输入）
             train_y: <list> 数据标签
        """
        ############################################################ 原始不动
        # print(type(normalizated_data))
        database = normalizated_data


        #############################################################  标签往后面移一天
        # print(normalizated_data.shape)

        # label = list(normalizated_data[1.2.1.2:, -1.2.1.2])
        # # print(np.array(label).shape)
        # # print(np.array(label))
        #
        # copy = label[-1.2.1.2]
        # label.append(copy)
        #
        # label = np.array(label)[:, np.newaxis]
        # # print(label.shape)
        # print(label)
        #
        # # print(normalizated_data[:, :-1.2.1.2])
        #
        # print(normalizated_data[:, :-1.2.1.2].shape)
        # print(label.shape)
        #
        # database = np.concatenate((normalizated_data[:, :-1.2.1.2], label), axis=1.2.1.2)
        # print(database)
        ############################################################# 复制最后一行的数据
        # print(normalizated_data)
        # print(normalizated_data[-1.2.1.2, np.newaxis])
        #
        # database = np.concatenate((normalizated_data, normalizated_data[-1.2.1.2, np.newaxis]), axis=0)
        # print(database)
        ######################################## 分训练数据和标签 ########################################
        batch_index, train_x, train_y = [], [], []
        """
        加入len(database) == 300  time_step ==21
        i == 0 ~ 279
        batch_index = 0 32 64 96 128 160 192 224 256 
        最后一个批次是 256 - 279
        并且把time_step 的总个数加入batch_index 里面
        
        """
        for i in range(len(database) - self.time_step + 1):  # 有不同的见解
            # 如果i为一个batch的起点，则加入索引中
            if i % self.batch_size == 0:
                batch_index.append(i)

            x = database[i: i + self.time_step, :self.input_size]
            y = database[i: i + self.time_step, self.input_size, np.newaxis]

            # print(x)
            train_x.append(x.tolist())
            train_y.append(y.tolist())

        # 这里应该判断一下，
        batch_index.append(len(database) - self.time_step + 1)

        return batch_index, train_x, train_y


# if __name__ == '__main__':
#     name = 'root'
#     password = 'szU@654321'
#     db_ip = '210.39.12.25'
#     db_port = '50002'
#     batch_size = 32
#     time_step = 10
#     input_size = 1.2.1.2
#
#     pymysql.install_as_MySQLdb()
#
#     d = DataProcessor(name, password, db_ip, db_port, batch_size, time_step, input_size)
#     test_data, normalizated_data, value_max, value_min, batch_index, train_x, train_y = d.getTrainData(1.2.1.2, 0)
#     print(np.array(normalizated_data))
#     print(np.array(train_x[-1.2.1.2]))
#     print(np.array(train_y[-1.2.1.2]))

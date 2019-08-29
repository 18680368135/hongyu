from DataAccessor import DataAccessor
import numpy as np
import pymysql
import pandas as pd


class DataProcessor(object):
    def __init__(self, batch_size, time_step, input_size, df):

        self.index = 0
        self.stock = 1

        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size

        self.df = df

    @staticmethod
    def peak_trough(data):
        """
        Effect:
            找出数据中的波峰与波谷
        :param
            data: <np.ndarray> 需要处理的数据
        :returns
            peak: <list> 波峰的索引
            trough: <list> 波谷的索引
        """
        # print(len(data))
        peak, trough = [], []
        size = len(data)
        i = 1
        if data[0] > data[1]:
            peak.append(0)
            while i < size - 1:
                while i < size - 1 and data[i] >= data[i + 1]:
                    i += 1
                save_value = i
                trough.append(i)
                while i < size - 1 and data[i] <= data[i + 1]:
                    i += 1
                if save_value != i:
                    peak.append(i)

        else:
            trough.append(0)
            while i < size - 1:
                while i < size - 1 and data[i] <= data[i + 1]:
                    i += 1
                save_value = i
                peak.append(i)
                while i < size - 1 and data[i] >= data[i + 1]:
                    i += 1
                if save_value != i:
                    trough.append(i)

        return peak, trough

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
        value_max = np.max(data, axis=0)
        value_min = np.min(data, axis=0)

        normalizated_data = (data - value_min) / (value_max - value_min)

        normalizated_data = np.nan_to_num(normalizated_data)

        return normalizated_data, value_max, value_min

    def trendLabeling(self, data, alpha1=1/2, beta1=1/2, alpha2=1/2, beta2=1/2):
        """
        Effect:
            根据道氏理论，找出区间段内的股票波动的特征
            四种情况
            1.2.1.2、如果先涨后跌，回撤幅度不得超过alpha1倍        判定为上涨
            2、如果先涨后跌，下跌幅度要超过上涨幅度的beta1倍     判定为下跌
            3、如果先跌后涨，上涨幅度超过下跌幅度的alpha2倍     判定为上涨
            4、如果先跌后涨，上涨幅度不超过下跌幅度的beta2倍     判定为下跌
        Parameters：
            data: 数据
            alpha, beta: 上涨、回撤比例
        Return：
            0（下降趋势） 1.2.1.2（震荡趋势） 2（上升趋势）
            3（上涨的次数大于下跌次数） 4（上涨的次数低于下跌的次数）
        """
        peak, trough = self.peak_trough(data)
        k = len(peak) if len(peak) < len(trough) else len(trough)

        t = 0
        if len(peak) == len(trough):
            t = 1

        cursor = 0
        trend = 1
        up, down, no = 0, 0, 0
        if peak[0] > trough[0]:
            while cursor < k - t:
                retrace = (data[peak[cursor]] - data[trough[cursor + 1]]) / (data[peak[cursor]] - data[trough[cursor]])
                if retrace < alpha1:
                    up += 1
                elif retrace > 1 + beta1:
                    down += 1
                else:
                    no += 1

                cursor += 1

        else:
            while cursor < k - t:
                retrace = (data[peak[cursor + 1]] - data[trough[cursor]]) / (data[peak[cursor]] - data[trough[cursor]])
                if retrace > 1 + alpha2:
                    up += 1
                elif retrace < beta2:
                    down += 1
                else:
                    no += 1

                cursor += 1

        if up == k:
            trend = 2
        elif down == k:
            trend = 0

        return trend

    def getLabelInfo(self, data):
        """
        Effect:
            将数据按照步长进行标记
        Param:
            data: <np.ndarray> 需要标记的数据
        Return:
            labels: <list> label的信息
        """
        data_to_label = []

        for start, end in zip(
            range(0, len(data) + 1),
            range(self.time_step, len(data) + 1)
        ):
            temp_ = data[start: end]
            data_to_label.append(temp_)

        data_to_label = np.array(data_to_label)
        labels = [self.trendLabeling(d) for d in data_to_label]

        # zero = 0
        # one = 0
        # two = 0
        # print(labels)
        # for i in labels:
        #     if i == 0:
        #         zero += 1.2.1.2
        #     elif i==2:
        #         two += 1.2.1.2
        #     else:
        #         one += 1.2.1.2
        # print(zero)
        # print(one)
        # print(two)

        return labels

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

        test_data = df.iloc[-days, :].values[1]
        df = df.iloc[:-days, :]

        return test_data, df

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
        label = list(normalizated_data[1:])

        copy = normalizated_data[-1]
        label.append(copy)

        database = np.concatenate((normalizated_data, label), axis=1)
        ######################################## 分训练数据和标签 ########################################
        batch_index, train_x, train_y = [], [], []

        for i in range(len(database) - self.time_step + 1):
            # 如果i为一个batch的起点，则加入索引中
            if i % self.batch_size == 0:
                batch_index.append(i)

            x = database[i: i + self.time_step, :self.input_size]
            y = database[i: i + self.time_step, self.input_size, np.newaxis]

            train_x.append(x.tolist())
            train_y.append(y.tolist())

        batch_index.append(len(database) - self.time_step + 1)

        return batch_index, train_x, train_y

    def getTestData(self, data):
        test_data = data[-30:, 0]

        test_x = []

        for i in range(len(test_data) - self.time_step + 1):
            x = test_data[i: i + self.time_step, np.newaxis]

            test_x.append(x.tolist())

        return test_x

    def choice_generate_data(self, generate_data_group, random_index):
        data = None

        for i in random_index:
            # print(generate_data_group[:, :, i, np.newaxis].shape)
            if data is None:
                data = generate_data_group[:, :, i, np.newaxis]

            else:
                data = np.concatenate((data, generate_data_group[:, :, i, np.newaxis]), axis=2)

        return data

    def getVerificationData(self, data):

        batch_index, verification_prediction, verfifcation_prevention = [], [], []

        for i in range(len(data) - self.time_step + 1):
            if i % self.batch_size == 0:
                batch_index.append(i)

            x1 = data[i: i + self.time_step, 0, np.newaxis]
            x2 = data[i: i + self.time_step, 1:]

            verification_prediction.append(x1.tolist())
            verfifcation_prevention.append(x2.tolist())

        batch_index.append(len(data) - self.time_step + 1)

        return batch_index, verification_prediction, verfifcation_prevention


if __name__ == '__main__':
    name = 'root'
    password = 'szU@654321'
    db_ip = '210.39.12.25'
    db_port = '50002'
    batch_size = 32
    time_step = 10
    input_size = 1

    pymysql.install_as_MySQLdb()

    d = DataProcessor(name, password, db_ip, db_port, batch_size, time_step, input_size)
    test_data, normalizated_data, value_max, value_min, batch_index, train_x, train_y = d.getTrainData(1, 0)
    print(np.array(normalizated_data))
    print(np.array(train_x[-1]))
    print(np.array(train_y[-1]))

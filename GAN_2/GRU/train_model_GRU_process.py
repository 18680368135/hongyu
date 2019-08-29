# encoding=utf-8
from GRU import GRU
import tensorflow as tf
import numpy as np
import os
import pandas as pd


class train_model():
    def __init__(self, timeStep, hiddenUnit, GeneratorInputSize, GeneratorOutputSize,
                 discriminatorInputSize, dim, c=0.01, learningRate=0.00001, k=1, epochs=50,
                 outputGraph=False, **kwargs):
        self.timeStep = timeStep
        self.hiddenUnit = hiddenUnit
        self.genInputSize = GeneratorInputSize
        self.genOutputSize = GeneratorOutputSize

        self.disInputSize = discriminatorInputSize
        self.dim = dim
        self.c = c
        self.lr = learningRate
        self.k = k
        self.epochs = epochs

        self.my_config = tf.ConfigProto()
        self.my_config.log_device_placement = False  # 输出设备和tensor详细信息
        self.my_config.gpu_options.allow_growth = True  # 基于运行需求分配显存(自动增长)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.buildModel()

        if outputGraph:
            with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
                with self.graph.as_default():
                    tf.summary.FileWriter('logs/', sess.graph)

    def build_lstm(self, generatorInput):
        gru = GRU(rnn_unit = self.hiddenUnit, input_size=self.genInputSize,
                               output_size = self.genOutputSize, X=generatorInput)
        # lstm.pred (None, 20, 1.2.1.2)
        # print('111')
        # print(lstm.pred)
        # print(lstm.pred[:, -1.2.1.2, tf.newaxis])
        # print(self.realData[:, :-1.2.1.2, :])

        # (None, 21, 1.2.1.2)
        # print(fakeData)
        return gru.pred

    def buildModel(self):
        self.lstmInput = tf.placeholder(tf.float32, [None, self.timeStep, self.genInputSize],
                                             name='gruInput')

        self.realData = tf.placeholder(tf.float32, [None, self.timeStep, self.genInputSize],
                                        name='realData')
        with tf.variable_scope('gru', reuse=False):
            self.predictValue = self.build_lstm(self.lstmInput)   # predictValue [32,20,1]

        self.squareLoss = tf.reduce_mean(
            tf.square(tf.reshape(self.realData, [-1]) - tf.reshape(self.predictValue, [-1])))  # 文中公式(4)

        self.Optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.squareLoss)


    def trainAndPredict(self, data, dataProcessor, day):
        """
        传入的需要参与到训练的数据,使得模型进行训练
        :param data:
        :param dataProcessor:
        :return:
        """
        # print(np.array(data))

        """
        indicators 去除最后一行之后为Generator的输入
        realData 为真实数据，作为Dis的一个输入
        """

        # print(data[-25:, :])
        # print(data.shape)
        batchIndex, indicators, realData = dataProcessor.getTrainData(data)
        # print(np.array(indicators[-1.2.1.2]))
        # print(np.array(realData[-1.2.1.2]))
        genInputs = np.array(indicators)[:, :, :]

        # print(np.array(batchIndex).shape)
        # print(np.array(indicators).shape)
        # print(np.array(realData).shape)
        # #
        # print(np.array(indicators[-1.2.1.2]))
        # print(genInputs[-1.2.1.2])
        # print(np.array(realData[-1.2.1.2]))

        testInput = np.array(indicators)[tf.newaxis, -1, :, :]
        # print(testInput)
        # print(testInput.shape)

        trainHist = {}

        trainHist['squareLoss'] = []

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
            with self.graph.as_default():
                sess.run(tf.global_variables_initializer())

                for epoch in range(self.epochs):
                    # print('Epoch %d of %d' % (epoch, self.epochs))
                    squareLoss = []

                    for step in range(len(batchIndex) - 1):
                        genIn = genInputs[batchIndex[step]: batchIndex[step + 1]]

                        real = np.array(realData[batchIndex[step]: batchIndex[step + 1]])

                        # print(genIn.shape)
                        # print(real.shape)
                        # initial G Cluster
                        optim, loss = sess.run([self.Optim, self.squareLoss], feed_dict={
                                            self.lstmInput: genIn,
                                            self.realData: real
                                            })


                        squareLoss.append(loss)
                    sqrloss = np.mean(squareLoss)
                    # print("day : %d epoch : %d, , loss : %f" % (day, epoch, sqrloss))
                    trainHist['squareLoss'].append(sqrloss)


            if not os.path.isdir('./loss_GRU/'):
                os.mkdir('./loss_GRU/')

            if not os.path.isdir('./loss_GRU/%d/' % day):
                os.mkdir('./loss_GRU/%d/' % day)


            dataframe = pd.DataFrame({'gru': trainHist['squareLoss']})
            dataframe.to_csv('./loss_GRU/%d/Loss.csv' % day, index=False, sep=',')


            # 至此模型的训练以及相关数据的保存已经完毕，接下来预测后一天的价格
            price = sess.run(self.predictValue,
                             feed_dict={self.lstmInput: testInput})

        return price


if __name__ == '__main__':
    para = dict(timeStep=21, hiddenUnit=32, GeneratorInputSize=13, GeneratorOutputSize=1,
                 discriminatorInputSize=1, dim=16, c=0.01, learningRate=0.00001, epochs=5, outputGraph=True)
    model = train_model(**para)
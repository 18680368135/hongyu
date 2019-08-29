# encoding=utf-8
from LSTM import LSTM
import tensorflow as tf
import numpy as np
import os
import pandas as pd


class GANFD():
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
        self.ncandi = 1  # 种群大小
        self.beta = 0.02
        self.nloss = 3  # 生成器三种loss类型
        self.loss = ['ls', 'minimax', 'trickLogD']


        with self.graph.as_default():
            self.buildModel()

        if outputGraph:
            with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
                with self.graph.as_default():
                    tf.summary.FileWriter('logs/', sess.graph)

    def generator(self, generatorInput):
        lstm = LSTM(rnn_unit = self.hiddenUnit, input_size = self.genInputSize,
                               output_size = self.genOutputSize, X=generatorInput)
        # lstm.pred (None, 20, 1.2.1.2)
        # print('111')
        # print(lstm.pred)
        # print(lstm.pred[:, -1.2.1.2, tf.newaxis])
        # print(self.realData[:, :-1.2.1.2, :])

        # (None, 21, 1.2.1.2)
        # print(fakeData)
        return lstm.pred

    def discriminator(self, disInput):
        with tf.name_scope('layer1'):
            conv1 = tf.layers.conv1d(disInput, filters=self.dim, kernel_size=5, strides=2,
                                     padding='SAME', name='conv1')

            conv1Lr = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1))
            # (None, 11, dim)
            # print(conv1Lr)

        with tf.name_scope('layer2'):
            conv2 = tf.layers.conv1d(conv1Lr, filters=self.dim * 2, kernel_size=5, strides=2,
                                     padding='SAME', name='conv2')

            conv2Lr = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2))
            # (None, 6, 2 * dim)
            # print(conv2Lr)

        with tf.name_scope('layer3'):
            conv3 = tf.layers.conv1d(conv2Lr, filters=self.dim * 4, kernel_size=5, strides=2,
                                        padding='valid', name='conv3')
            conv3Lr = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3))
            # (None, 1.2.1.2, 4 * dim)
            # print(conv3Lr)

        with tf.name_scope('fc1'):
            fc1 = tf.layers.dense(conv3Lr, self.dim * 2, activation=tf.nn.leaky_relu, name='fc1')
            # (None, 1.2.1.2, dim * 2)
            # print(fc1)

        with tf.name_scope('fc2'):
            fc2 = tf.layers.dense(fc1, self.dim, activation=tf.nn.leaky_relu, name='fc2')
            # (None, 1.2.1.2, dim)
            # print(fc2)

        with tf.name_scope('output'):
            output = tf.nn.sigmoid(tf.layers.dense(fc2, 1), name='output')
            # (None, 1.2.1.2, 1.2.1.2)
            # print(output)

        return output

    def buildModel(self):
        self.genInput = tf.placeholder(tf.float32, [None, self.timeStep - 1, self.genInputSize],
                                             name='GeneratorInput')

        self.realData = tf.placeholder(tf.float32, [None, self.timeStep, self.disInputSize],
                                        name='realData')

        with tf.variable_scope("Discriminator", reuse=False):
            self.dLogitsReal = self.discriminator(self.realData)

        # self.pReal = tf.reduce_mean(self.dLogitsReal, name='realLoss')
        self.pReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsReal, labels=tf.ones_like(self.dLogitsReal)), name='realLoss')

        with tf.variable_scope('Generator', reuse=False):
            self.predictValue = self.generator(self.genInput)

        self.fakeData = tf.concat((self.realData[:, :-1, :], self.predictValue[:, -1, tf.newaxis]), axis=1, name='fakeData')
            # print(self.predictValue)
        with tf.variable_scope("Discriminator", reuse=True):
            self.dLogitsFake = self.discriminator(self.fakeData)

        # self.pGen = tf.reduce_mean(self.dLogitsFake, name='fakeLoss')

        self.pGen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dLogitsFake, labels=tf.zeros_like(self.dLogitsFake)), name='fakeLoss')


        self.squareLoss = tf.reduce_mean(
            tf.square(tf.reshape(self.realData, [-1]) - tf.reshape(self.fakeData, [-1])))  # 文中公式(4)

        directLoss = tf.reduce_mean(tf.abs(
            tf.sign(self.fakeData[:, -1, :] - self.realData[:, -2, :]) -            # 文中公式(5)
            tf.sign(self.realData[:, -1, :] - self.realData[:, -2, :])))            # 这一段代码好好检查一下逻辑有没有错误


        self.dLoss = self.pReal + self.pGen

        self.gLoss_trickLogD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsFake, labels=tf.ones_like(self.dLogitsFake))) + self.squareLoss + directLoss
        self.gLoss_minimax = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsFake, labels=tf.zeros_like(self.dLogitsFake))) + self.squareLoss + directLoss
        self.gLoss_ls = tf.reduce_mean(tf.squared_difference(self.dLogitsFake, tf.ones_like(self.dLogitsFake))) + self.squareLoss + directLoss

        TVars = tf.trainable_variables()

        DVars = [var for var in TVars if var.name.startswith('Discriminator')]
        GVars = [var for var in TVars if var.name.startswith('Generator')]

        self.clipD = [p.assign(tf.clip_by_value(p, -self.c, self.c)) for p in DVars]


        self.dOptim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.dLoss, var_list=DVars)


        # self.gOptim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss, var_list=GVars)

        self.gOptim_ls = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_ls, var_list=GVars)
        self.gOptim_minimax = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_minimax, var_list=GVars)
        self.gOptim_trickLogD = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_trickLogD, var_list=GVars)

        # FD_ls = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.gLoss_ls, var_list=DVars)
        # FD_minimax = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.gLoss_minimax, var_list=DVars)
        # FD_trickLogD = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.gLoss_trickLogD, var_list=DVars)

        FD_ls = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.dLoss, var_list=DVars)
        FD_minimax = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.dLoss, var_list=DVars)
        FD_trickLogD = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.dLoss, var_list=DVars)

        self.Fd_score_ls = self.beta * tf.log(sum(tf.reduce_sum(tf.square(x))for x in FD_ls))
        self.Fd_score_minimax = self.beta * tf.log(sum(tf.reduce_sum(tf.square(x)) for x in FD_minimax))
        self.Fd_score_trickLogD = self.beta * tf.log(sum(tf.reduce_sum(tf.square(x)) for x in FD_trickLogD))


    def _min_fitness(self, F_ls, F_minimax, F_trickLogD):
        if F_ls < F_minimax:
            if F_ls < F_trickLogD:
                return 'ls'
            else:
                return 'trickLogD'
        else :
            if F_minimax > F_trickLogD:
                return 'trickLogD'
            else:
                return 'minimax'


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
        genInputs = np.array(indicators)[:, :-1, :]

        # print(np.array(batchIndex).shape)
        # print(np.array(indicators).shape)
        # print(np.array(realData).shape)
        # #
        # print(np.array(indicators[-1.2.1.2]))
        # print(genInputs[-1.2.1.2])
        # print(np.array(realData[-1.2.1.2]))

        testInput = np.array(indicators)[tf.newaxis, -1, 1:, :]
        # print(testInput)
        # print(testInput.shape)

        trainHist = {}
        trainHist['DLoss'] = []
        trainHist['GLoss'] = []
        trainHist['pReal'] = []
        trainHist['pFake'] = []
        trainHist['squareLoss'] = []
        trainHist['bestGenLoss'] = []



        iterations = 1
        for i in range(self.ncandi):
            with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
                with self.graph.as_default():

                    sess.run(tf.global_variables_initializer())
                    for epoch in range(self.epochs):
                        print('Epoch %d of %d' % (epoch + 1, self.epochs))
                        GLosses, DLosses = [], []
                        pReal, pFake = [], []
                        squareLoss = []
                        genLossTYpe = []

                        for step in range(len(batchIndex) - 1):
                            genIn = genInputs[batchIndex[step]: batchIndex[step + 1]]

                            real = np.array(realData[batchIndex[step]: batchIndex[step + 1]])

                            # print(genIn.shape)
                            # print(real.shape)

                            _, dLossVal, pRealVal, pFakeVal = sess.run(
                                [self.dOptim, self.dLoss, self.pReal, self.pGen],
                                feed_dict={
                                    self.genInput: genIn,
                                    self.realData: real
                                })

                            # sess.run(self.clipD)

                            if (iterations - 1) % self.k == 0:
                                for j in range(0, self.nloss):
                                    if self.loss[j] == 'ls':
                                        gLossVal, _, fd_score = sess.run(
                                            [self.gLoss_ls, self.squareLoss, self.Fd_score_ls],
                                            feed_dict={
                                                self.genInput: genIn,
                                                self.realData: real
                                            })
                                        F_ls = gLossVal - fd_score
                                    elif self.loss[j] == 'minimax':
                                        gLossVal, _, fd_score = sess.run(
                                            [self.gLoss_minimax, self.squareLoss, self.Fd_score_minimax],
                                            feed_dict={
                                                self.genInput: genIn,
                                                self.realData: real
                                            })
                                        F_minimax = gLossVal - fd_score
                                    elif self.loss[j] == 'trickLogD':
                                        gLossVal, _, fd_score = sess.run(
                                            [self.gLoss_trickLogD, self.squareLoss, self.Fd_score_trickLogD],
                                            feed_dict={
                                                self.genInput: genIn,
                                                self.realData: real
                                            })
                                        F_trickLogD = gLossVal - fd_score
                                    print("gLossVal: %f, fd_score: %f" % (gLossVal, fd_score))

                                print("F_ls: %f,F_minimax: %f,F_trickLogD: %f" % (F_ls, F_minimax, F_trickLogD))

                                best_gen = self._min_fitness(F_ls, F_minimax, F_trickLogD)
                                if best_gen == 'ls':
                                    _, sqrLoss = sess.run([self.gOptim_ls, self.squareLoss], feed_dict={self.genInput: genIn, self.realData: real})
                                elif best_gen == 'minimax':
                                    _, sqrLoss = sess.run([self.gOptim_minimax, self.squareLoss], feed_dict={self.genInput: genIn, self.realData: real})
                                elif best_gen == 'trickLogD':
                                    _, sqrLoss = sess.run([self.gOptim_trickLogD, self.squareLoss], feed_dict={self.genInput: genIn, self.realData: real})

                            GLosses.append(gLossVal)
                            DLosses.append(dLossVal)
                            pReal.append(pRealVal)
                            pFake.append(pFakeVal)
                            squareLoss.append(sqrLoss)
                            genLossTYpe.append(best_gen)

                            iterations += 1

                        trainHist['DLoss'].append(np.mean(DLosses))
                        trainHist['GLoss'].append(np.mean(DLosses))
                        trainHist['pReal'].append(np.mean(pReal))
                        trainHist['pFake'].append(np.mean(pFake))
                        trainHist['squareLoss'].append(np.mean(squareLoss))
                        trainHist['bestGenLoss'].append(genLossTYpe)

                if not os.path.isdir('./loss/'):
                    os.mkdir('./loss/')

                if not os.path.isdir('./loss/%d/' % day):
                    os.mkdir('./loss/%d/' % day)

                dataframe = pd.DataFrame({'real': trainHist['pReal'], 'fake': trainHist['pFake']})
                dataframe.to_csv('./loss/%d/realFakeLoss.csv' % day, index=False, sep=',')

                dataframe = pd.DataFrame({'D': trainHist['DLoss'], 'G': trainHist['GLoss']})
                dataframe.to_csv('./loss/%d/GANLoss.csv' % day, index=False, sep=',')
                dataframe = pd.DataFrame({'D': trainHist['squareLoss']})
                dataframe.to_csv('./loss/%d/Loss.csv' % day, index=False, sep=',')

                dataframe = pd.DataFrame({'G': trainHist['bestGenLoss']})
                dataframe.to_csv('./loss/%d/genLossType.csv' % day, index=False, sep=',')

                # 至此模型的训练以及相关数据的保存已经完毕，接下来预测后一天的价格
                price = sess.run(self.predictValue,
                                 feed_dict={self.genInput: testInput})

        return price


if __name__ == '__main__':
    para = dict(timeStep=21, hiddenUnit=32, GeneratorInputSize=13, GeneratorOutputSize=1,
                 discriminatorInputSize=1, dim=16, c=0.01, learningRate=0.00001, epochs=5, outputGraph=True)
    model = GANFD(**para)
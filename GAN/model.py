# encoding=utf-8
from LSTM import LSTM
import tensorflow as tf
import numpy as np
import os
import pandas as pd


class GAN():
    def __init__(self, timeStep, dim, lr, c, epoch,
                 predictionInputSize, predictionOutputSize, predictionRnnUnit, predictionFc,
                 preventionInputSize, preventionOutputSize, preventionRnnUnit, preventionFc,
                 fc2, outputGraph=False):
        self.timeStep = timeStep

        self.predictionInputSize = predictionInputSize
        self.predictionOutputSize = predictionOutputSize
        self.predictionRnnUnit = predictionRnnUnit
        self.predictionFc = predictionFc

        self.preventionInputSize = preventionInputSize
        self.preventionOutputSize = preventionOutputSize
        self.preventionRnnUnit = preventionRnnUnit
        self.preventionFc = preventionFc

        self.fc2 = fc2
        self.dim = dim
        self.lr = lr
        self.c = c
        self.epoch = epoch

        self.my_config = tf.ConfigProto()
        self.my_config.log_device_placement = False  # 输出设备和tensor详细信息
        self.my_config.gpu_options.allow_growth = True  # 基于运行需求分配显存(自动增长)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.buildModel()

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
            with self.graph.as_default():
                if outputGraph:
                    tf.summary.FileWriter('logs/', sess.graph)

    def Generator(self, predictionInput, preventionInput):
        """
        构建生成器
        :param predictionInput: <tf.placeholder> prediction模块的输入
        :param preventionInput: <tf.placeholder> prevention模块的输入
        :return:
            GeneratorOutput：拼接了prediction的输入的“假”数据
            predResult： 通过Mod模型预测的下一天的结果
        """
        with tf.variable_scope('Prediction'):
            lstm_prediction = LSTM(self.predictionRnnUnit, self.predictionInputSize, self.predictionOutputSize,
                                   predictionInput)
            predictFc1 = tf.layers.dense(lstm_prediction.pred, self.predictionFc, name='predictionFc')
            predictFc1 = tf.nn.leaky_relu(tf.layers.batch_normalization(predictFc1))
            # (None, timeStep, predictionFc)
        print(predictFc1)

        # with tf.variable_scope('Prevention'):
        #     lstm_prevention = LSTM(self.preventionRnnUnit, self.preventionInputSize, self.preventionOutputSize,
        #                            preventionInput)
        #     preventionFc1 = tf.layers.dense(lstm_prevention.pred, self.preventionFc, name='preventionFc')
            # (None, timeStep, preventionFc)

        # with tf.variable_scope('concatenation'):
        #     output = tf.concat([predictFc1, preventionFc1], axis=2, name='concat')
            # (None, timeStep, predictionFc + preventionFc)

        # with tf.variable_scope('fc2'):
        #     fc = tf.layers.dense(output, self.fc2, name='fc2')
            # (None, timeStep, fc2)

        with tf.variable_scope('output'):
            predResult = tf.layers.dense(predictFc1, 1, name='predictResult')
            predResult = tf.nn.leaky_relu(tf.layers.batch_normalization(predResult))
            # (None, timeStep, 1.2.1.2)
        print(predResult)
        with tf.variable_scope('GeneratorOutput'):
            GeneratorOutput = tf.concat((self.predictionInput, predResult[:, -1, np.newaxis]),
                                             axis=1, name='output')
            GeneratorOutput = tf.nn.tanh(GeneratorOutput)
            # (None, timeStep + 1.2.1.2, 1.2.1.2)
        print(GeneratorOutput)
        return GeneratorOutput, predResult

    def Discriminator(self, inputDis):
        """
        构建判别器
        :param inputDis: <tf.placeholder> 判别器的输入
        :return: Discriminator的输出
        """
        with tf.name_scope('layer1'):
            conv1 = tf.layers.conv1d(inputDis, filters=self.dim, kernel_size=5, strides=2,
                                     padding='SAME', name='conv1')
            conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1))
        # (None, 10, dim)
        print(conv1)

        with tf.name_scope('layer2'):
            conv2 = tf.layers.conv1d(conv1, filters=self.dim * 2, kernel_size=5, strides=2,
                                     padding='SAME', name='conv2')
            conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2))
        # (None, 5, 2*dim)    print(conv2)
        print(conv2)

        with tf.name_scope('layer3'):
            DisOutput = tf.layers.conv1d(conv2, filters=1, kernel_size=5, strides=2,
                                        padding='valid', name='conv3')
        # (None, 1.2.1.2, 1.2.1.2)
        print(DisOutput)
        return DisOutput

    def buildModel(self):
        """
        构建好整个GAN的模型架构
        """
        self.predictionInput = tf.placeholder(tf.float32, [None, self.timeStep - 1, self.predictionInputSize])
        self.preventionInput = tf.placeholder(tf.float32, [None, self.timeStep - 1, self.preventionInputSize])     # 本质是z

        self.realData = tf.placeholder(tf.float32, [None, self.timeStep, 1])

        with tf.variable_scope('Discriminator', reuse=False):
            self.dLogitsReal = self.Discriminator(self.realData)

        self.pReal = tf.reduce_mean(self.dLogitsReal, name='real_loss')

        with tf.variable_scope('Generator', reuse=False):
            self.fakeData, self.predictValue = self.Generator(self.predictionInput, self.preventionInput)
        with tf.variable_scope('Discriminator', reuse=True):
            self.dLogitsFake = self.Discriminator(self.fakeData)

        self.pGen = tf.reduce_mean(self.dLogitsFake, name='fake_loss')

        self.dLoss = - (self.pReal - self.pGen)
        self.gLoss = - self.pGen

        TVars = tf.trainable_variables()

        DVars = [var for var in TVars if var.name.startswith('Discriminator')]
        GVars = [var for var in TVars if var.name.startswith('Generator')]

        self.clipD = [p.assign(tf.clip_by_value(p, -self.c, self.c)) for p in DVars]

        self.dOptim = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.dLoss, var_list=DVars)
        self.gOptim = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.gLoss, var_list=GVars)


    def train(self, batchIndex, trainX):

        trainHist = {}
        trainHist['DLoss'] = []
        trainHist['GLoss'] = []
        trainHist['pReal'] = []
        trainHist['pFake'] = []

        iterations = 1
        k = 1

        partOfTrainX = np.array(trainX)[:, :-1, :]

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
            with self.graph.as_default():

                saver = tf.train.Saver()

                if os.path.isfile('./model/checkpoint'):
                    module_file = tf.train.latest_checkpoint('./model/')
                    saver.restore(sess, module_file)
                    print('Reload model')
                else:
                    print('Setup new model')
                    sess.run(tf.global_variables_initializer())

                for epoch in range(self.epoch):

                    GLosses, DLosses = [], []
                    pReal, pFake = [], []

                    for step in range(len(batchIndex) - 1):
                        realData = np.array(trainX[batchIndex[step]: batchIndex[step + 1]])
                        predictionInput = partOfTrainX[batchIndex[step]: batchIndex[step + 1]]

                        # print(realData.shape)
                        # print(predictionInput.shape)

                        batchSize = realData.shape[0]

                        Zs = np.random.normal(-1, 1, size=[batchSize, self.timeStep - 1, self.preventionInputSize])

                        _, dLossVal, gLossVal, pRealVal, pFakeVal = sess.run(
                            [self.dOptim, self.dLoss, self.gLoss, self.pReal, self.pGen], feed_dict={
                                self.predictionInput: predictionInput,
                                self.preventionInput: Zs,
                                self.realData: realData
                            })

                        sess.run(self.clipD)

                        if (iterations - 1) % k == 0:
                            _, gLossVal, pRealVal, pFakeVal = sess.run(
                                [self.gOptim, self.gLoss, self.pReal, self.pGen], feed_dict={
                                    self.predictionInput: predictionInput,
                                    self.preventionInput: Zs,
                                    self.realData: realData
                                })

                        GLosses.append(gLossVal)
                        DLosses.append(dLossVal)
                        pReal.append(pRealVal)
                        pFake.append(pFakeVal)

                        iterations += 1

                    trainHist['DLoss'].append(np.mean(DLosses))
                    trainHist['GLoss'].append(np.mean(DLosses))
                    trainHist['pReal'].append(np.mean(pReal))
                    trainHist['pFake'].append(np.mean(pFake))

                if not os.path.isdir('./model/'):
                    os.mkdir('./model')

                saver.save(sess, os.path.join(os.getcwd(), './model/model.ckpt'))

                if not os.path.isdir('./loss2/'):
                    os.mkdir('./loss2/')

                dataframe = pd.DataFrame({'real': trainHist['pReal'], 'fake': trainHist['pFake']})
                dataframe.to_csv('./loss2/realFakeLoss.csv', index=False, sep=',')

                dataframe = pd.DataFrame({'D': trainHist['DLoss'], 'G': trainHist['GLoss']})
                dataframe.to_csv('./loss2/GANLoss.csv', index=False, sep=',')


    def predict(self, batchIndex, testX):

        partOfTestX = np.array(testX)[:, :-1, :]

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
            with self.graph.as_default():
                saver = tf.train.Saver()

                moduel_file = tf.train.latest_checkpoint('./model/')
                saver.restore(sess, moduel_file)

                testPredict = []

                for step in range(len(batchIndex) - 1):

                    predictionInput = partOfTestX[batchIndex[step]: batchIndex[step + 1]]

                    # print(realData.shape)
                    # print(predictionInput.shape)

                    batchSize = predictionInput.shape[0]

                    Zs = np.random.normal(-1, 1, size=[batchSize, self.timeStep - 1, self.preventionInputSize])

                    predictValue = sess.run(self.predictValue, feed_dict={
                        self.predictionInput: predictionInput,
                        self.preventionInput: Zs,
                    })

                predict = predictValue
                testPredict.extend(predict)

        return testPredict[-1]


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    predictionInput = tf.placeholder(tf.float32, [10, 15, 1])
    preventionInput = tf.placeholder(tf.float32, [10, 15, 5])
    predictionInputSize = 1
    predictionOutputSize = 3
    predictionRnnUnit = 32
    predictionFc = 2
    preventionInputSize = 5
    preventionOutputSize = 3
    preventionRnnUnit = 32
    preventionFc = 2
    fc2 = 3
    #
    #
    # generator = Generator(predictionInput, preventionInput, 15,
    #              predictionInputSize, predictionOutputSize, predictionRnnUnit, predictionFc,
    #              preventionInputSize, preventionOutputSize, preventionRnnUnit, preventionFc,
    #              fc2)

    # inputDis = tf.placeholder(tf.float32, [None, 21, 1.2.1.2])
    # Discriminator(inputDis, 16)

    gan = GAN(20, 16, 0.0001, 0.01, 50,
        predictionInputSize, predictionOutputSize, predictionRnnUnit, predictionFc,
        preventionInputSize, preventionOutputSize, preventionRnnUnit, preventionFc,
        fc2)


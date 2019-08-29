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
        self.KD = 3  # of discrim updates for each gen update
        self.KG = 1  # of discrim updates for each gen update
        self.ncandi = 1  # 种群大小
        self.beta = 0.02
        self.nloss = 3  # 生成器三种loss类型
        self.loss = ['minimax', 'ls', 'trickLogD']



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

        self.realOut = tf.reduce_mean(self.dLogitsReal)
        # self.pReal = tf.reduce_mean(self.dLogitsReal, name='realLoss')
        self.pReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsReal, labels=tf.ones_like(self.dLogitsReal)), name='realLoss')

        with tf.variable_scope('Generator', reuse=False):
            self.predictValue = self.generator(self.genInput)

        self.fakeData = tf.concat((self.realData[:, :-1, :], self.predictValue[:, -1, tf.newaxis]), axis=1, name='fakeData')
            # print(self.predictValue)
        with tf.variable_scope("Discriminator", reuse=True):
            self.dLogitsFake = self.discriminator(self.fakeData)

        self.fakeOut = tf.reduce_mean(self.dLogitsFake)
        # self.pGen = tf.reduce_mean(self.dLogitsFake, name='fakeLoss')

        self.pGen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dLogitsFake, labels=tf.zeros_like(self.dLogitsFake)), name='fakeLoss')

        self.squareLoss = tf.reduce_mean(
            tf.square(tf.reshape(self.realData, [-1]) - tf.reshape(self.fakeData, [-1])))  # 文中公式(4)

        self.directLoss = tf.reduce_mean(tf.abs(
            tf.sign(self.fakeData[:, -1, :] - self.realData[:, -2, :]) -            # 文中公式(5)
            tf.sign(self.realData[:, -1, :] - self.realData[:, -2, :])))            # 这一段代码好好检查一下逻辑有没有错误


        self.dLoss = self.pReal + self.pGen

        self.gLoss_trickLogD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsFake, labels=tf.ones_like(self.dLogitsFake))) + self.squareLoss + self.directLoss
        self.gLoss_minimax = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsFake, labels=tf.zeros_like(self.dLogitsFake))) + self.squareLoss + self.directLoss
        self.gLoss_ls = tf.reduce_mean(tf.squared_difference(self.dLogitsFake, tf.ones_like(self.dLogitsFake))) + self.squareLoss + self.directLoss

        TVars = tf.trainable_variables()
        self.GVars = [var for var in TVars if var.name.startswith('Generator')]
        self.DVars = [var for var in TVars if var.name.startswith('Discriminator')]

        self.Gweight = []
        self.Gweight_value = []

        for i in range(len(self.GVars)):
            self.Gweight.append(tf.placeholder(tf.float32, self.GVars[i].shape.as_list()))
            self.Gweight_value.append(tf.assign(self.GVars[i], self.Gweight[i]))

        self.clipD = [p.assign(tf.clip_by_value(p, -self.c, self.c)) for p in self.DVars]

        self.dOptim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.dLoss, var_list=self.DVars)

        self.gOptim_ls = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_ls, var_list=self.GVars)
        self.gOptim_minimax = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_minimax, var_list=self.GVars)
        self.gOptim_trickLogD = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss_trickLogD, var_list=self.GVars)



        # 计算FD分数
        self.FD = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(loss=self.dLoss, var_list=self.DVars)
        self.Fd_score = self.beta * tf.log(sum(tf.reduce_sum(tf.square(x))for x in self.FD))

        # self.initParam = tf.global_variables_initializer()

    def create_G(self, loss_type=None):

        #  trickLogD 和 minimax 到底哪个是负值，需要验证一下
        if loss_type == 'trickLogD':
            self.gLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dLogitsFake, labels=tf.ones_like(self.dLogitsFake))) + self.squareLoss + self.directLoss
        elif loss_type == 'minimax':
            self.gLoss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dLogitsFake, labels=tf.zeros_like(self.dLogitsFake))) + self.squareLoss + self.directLoss
        elif loss_type == 'ls':
            self.gLoss = tf.reduce_mean(tf.squared_difference(self.dLogitsFake, tf.ones_like(self.dLogitsFake))) \
                             + self.squareLoss + self.directLoss

        self.gOptim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gLoss, var_list=self.GVars)

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


        gen_new_params = []
        gen_tem_param = []
        memory = 0
        n_updates = 0

        iterations = 1

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
            with self.graph.as_default():
                self.create_G(loss_type=self.loss[0])
                for i in range(1, 31):
                    sess.run(tf.global_variables_initializer())
                    Gvar_value = sess.run(self.GVars)

                    gen_tem_param.append(Gvar_value)
                    # print(type(gen_tem_param[0]))
                    if i % 10 == 0:
                        # print(type(np.mean(gen_tem_param, axis=0)[0]))
                        # print(len(np.mean(gen_tem_param, axis=0)[0]))
                        # print(len(np.mean(gen_tem_param, axis=0)))
                        # a = sess.run(tf.convert_to_tensor(np.mean(gen_tem_param, axis=0)))
                        gen_new_params.append(np.mean(gen_tem_param, axis=0))
                        gen_tem_param = []


                for epoch in range(self.epochs):
                    print('Epoch %d of %d' % (epoch + 1, self.epochs))
                    GLosses, DLosses = [], []
                    pReal, pFake = [], []
                    squareLoss = []

                    for step in range(len(batchIndex) - 1):
                        genIn = genInputs[batchIndex[step]: batchIndex[step + 1]]

                        real = np.array(realData[batchIndex[step]: batchIndex[step + 1]])

                        # print(genIn.shape)
                        # print(real.shape)
                        # initial G Cluster
                        if epoch + n_updates == 0:
                            for can_i in range(self.ncandi):
                                # self.create_G(loss_type=self.loss[can_i%self.nloss])
                                for _ in range(0, self.KG):
                                    _, gLossVal, pFakeVal, sqrLoss = sess.run(
                                        [self.gOptim_trickLogD, self.gLoss,
                                         self.pGen, self.squareLoss],
                                        feed_dict={
                                            self.genInput: genIn,
                                            self.realData: real
                                            })
                        else:
                            gen_old_params = gen_new_params
                            for can_i in range(0, self.ncandi):
                                for type_i in range(self.nloss):
                                    sess.run([self.Gweight_value[0],
                                              self.Gweight_value[1],
                                              self.Gweight_value[2],
                                              self.Gweight_value[3]],
                                             feed_dict={self.Gweight[0]: gen_old_params[type_i][0],
                                                        self.Gweight[1]: gen_old_params[type_i][1],
                                                        self.Gweight[2]: gen_old_params[type_i][2],
                                                        self.Gweight[3]: gen_old_params[type_i][3]})

                                    # sess.run(self.Gweight_value[i], feed_dict={self.Gweight[i]: gen_old_params[type_i][i]})

                                    if self.loss[type_i] == 'trickLogD':
                                        _ = sess.run([self.gOptim_trickLogD], feed_dict={self.genInput: genIn,
                                                                                         self.realData: real})
                                    elif self.loss[type_i] == 'minimax':
                                        _ = sess.run([self.gOptim_minimax], feed_dict={self.genInput: genIn,
                                                                                       self.realData: real})
                                    elif self.loss[type_i] == 'ls':
                                        _ = sess.run([self.gOptim_ls], feed_dict={self.genInput: genIn,
                                                                                  self.realData: real})
                                    # 计算适应度函数值

                                    fr_score, fd_score, dlossvalue, gen_new_params[type_i] = \
                                        sess.run([self.fakeOut, self.Fd_score, self.dLoss, self.GVars],
                                                 feed_dict={
                                                     self.genInput: genIn,
                                                     self.realData: real
                                                 })
                                    print("Fq : %r, fd : %r, dloss : %r" % (fr_score, fd_score, dlossvalue))
                                    fit = fr_score - fd_score
                                    if can_i * self.nloss + type_i < self.ncandi:
                                        idx = can_i * self.nloss + type_i
                                        fitness[idx] = fit
                                        fake_rate[idx] = fr_score
                                        memory = 0
                                        # generate_out_old[idx*32:(idx+1.2)*32,:,:,:] = generate_out
                                    else:
                                        fit_com = fitness - fit
                                        if (type_i == 1) and (min(fit_com) < 0):
                                            ids_replace = np.where(fit_com == min(fit_com))
                                            idr = ids_replace[0][0]

                                            fitness[idr] = fit
                                            fake_rate[idr] = fr_score
                                            memory = 1
                                        if (type_i == 2) and (min(fit_com) < 0):
                                            ids_replace = np.where(fit_com == min(fit_com))
                                            idr = ids_replace[0][0]

                                            fitness[idr] = fit
                                            fake_rate[idr] = fr_score
                                            memory = 2
                                        # print(idr)
                                print("epoch: %d,fake_rate:%r,fitness:%r" % (epoch, fake_rate, fitness))
                            sess.run([self.Gweight_value[0],
                                      self.Gweight_value[1],
                                      self.Gweight_value[2],
                                      self.Gweight_value[3]],
                                     feed_dict={self.Gweight[0]: gen_old_params[memory][0],
                                                self.Gweight[1]: gen_old_params[memory][1],
                                                self.Gweight[2]: gen_old_params[memory][2],
                                                self.Gweight[3]: gen_old_params[memory][3]})
                        # for i in range(len(self.Gweight)):
                        #     sess.run(self.Gweight_value[i], feed_dict={self.Gweight[i]: gen_new_params[memory][i]})
                        gLossVal, pFakeVal, sqrLoss = sess.run([self.gLoss, self.pGen, self.squareLoss],
                                                               feed_dict={
                                                                    self.genInput: genIn,
                                                                    self.realData: real
                                                                })

                        # train Discriminator
                        _, dLossVal, gLossVal, pRealVal = sess.run([self.dOptim, self.dLoss, self.gLoss, self.pReal],
                                     feed_dict={
                                         self.genInput: genIn,
                                         self.realData: real
                                     })
                        for i in range(0, self.ncandi):
                            tr, fr, fd = sess.run([self.realOut, self.fakeOut, self.Fd_score],
                                                  feed_dict={
                                                      self.genInput: genIn,
                                                      self.realData: real
                                                  })
                            if i == 0:
                                fake_rate = np.array([fr])
                                fitness = np.array([0.])
                                real_rate = np.array([tr])
                                FDL = np.array([fd])
                            else:
                                fake_rate = np.append(fake_rate, fr)
                                fitness = np.append(fitness, [0.])
                                real_rate = np.append(real_rate, tr)
                                FDL = np.append(FDL, fd)

                        print("fake_rate : %r, FDL :%r" % (fake_rate, FDL))
                        n_updates += 1

                        # sess.run(self.clipD)

                        # if (iterations - 1.2) % self.k == 0:
                        #     gLossVal,  pFakeVal, sqrLoss = sess.run(
                        #         [self.gLoss,  self.pGen, self.squareLoss],
                        #         feed_dict={
                        #             self.genInput: genIn,
                        #             self.realData: real
                        #         })

                        GLosses.append(gLossVal)
                        DLosses.append(dLossVal)
                        pReal.append(pRealVal)
                        pFake.append(pFakeVal)
                        squareLoss.append(sqrLoss)


                        iterations += 1

                    trainHist['DLoss'].append(np.mean(DLosses))
                    trainHist['GLoss'].append(np.mean(GLosses))
                    trainHist['pReal'].append(np.mean(pReal))
                    trainHist['pFake'].append(np.mean(pFake))
                    trainHist['squareLoss'].append(np.mean(squareLoss))


            if not os.path.isdir('./loss2/'):
                os.mkdir('./loss2/')

            if not os.path.isdir('./loss2/%d/' % day):
                os.mkdir('./loss2/%d/' % day)

            dataframe = pd.DataFrame({'real': trainHist['pReal'], 'fake': trainHist['pFake']})
            dataframe.to_csv('./loss2/%d/realFakeLoss.csv' % day, index=False, sep=',')

            dataframe = pd.DataFrame({'D': trainHist['DLoss'], 'G': trainHist['GLoss']})
            dataframe.to_csv('./loss2/%d/GANLoss.csv' % day, index=False, sep=',')
            dataframe = pd.DataFrame({'D': trainHist['squareLoss']})
            dataframe.to_csv('./loss2/%d/Loss.csv' % day, index=False, sep=',')


            # 至此模型的训练以及相关数据的保存已经完毕，接下来预测后一天的价格
            price = sess.run(self.predictValue,
                             feed_dict={self.genInput: testInput})

        return price


if __name__ == '__main__':
    para = dict(timeStep=21, hiddenUnit=32, GeneratorInputSize=13, GeneratorOutputSize=1,
                 discriminatorInputSize=1, dim=16, c=0.01, learningRate=0.00001, epochs=5, outputGraph=True)
    model = GANFD(**para)
# encoding=utf-8
from BiLSTM import BiLSTM
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from random import shuffle


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
        self.beta = 0.00001
        self.nloss = 3  # 生成器三种loss类型
        self.loss = ['trickLogD', 'minimax', 'ls']



        with self.graph.as_default():
            self.buildModel()

        if outputGraph:
            with tf.Session(graph=self.graph, config=self.my_config).as_default() as sess:
                with self.graph.as_default():
                    tf.summary.FileWriter('logs/', sess.graph)

    def get_all_step_index(self, batchLen):
        id_index = []
        for epoch in range(self.epochs):
            id = [i for i in range(batchLen)]
            shuffle(id)
            id_index.append(id)
        id_index = np.reshape(id_index, -1)
        id_len = len(id_index)
        id_len_reminder = id_len % 3
        id_len_division = int(id_len / 3)
        if (id_len_reminder == 0):
            step_index = np.reshape(id_index, (id_len_division, 3)).tolist()
        else:
            first_id_index = id_index[0: id_len_reminder]
            step_index = np.reshape(id_index[id_len_reminder:], (id_len_division, 3)).tolist()
            step_index.insert(0, first_id_index.tolist())

        return step_index, len(step_index)

    def generator(self, generatorInput):
        bilstm = BiLSTM(rnn_unit = self.hiddenUnit, input_size = self.genInputSize,
                               output_size = self.genOutputSize, X=generatorInput)
        # lstm.pred (None, 20, 1.2.1.2)
        # print('111')
        # print(lstm.pred)
        # print(lstm.pred[:, -1.2.1.2, tf.newaxis])
        # print(self.realData[:, :-1.2.1.2, :])

        # (None, 21, 1.2.1.2)
        # print(fakeData)
        return bilstm.pred

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

        self.pReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsReal, labels=tf.ones_like(self.dLogitsReal)), name='realLoss')

        # self.pReal = -tf.reduce_mean(tf.multiply(tf.ones_like(self.dLogitsReal), tf.log(self.dLogitsReal)) +
        #                              tf.multiply((1.-tf.ones_like(self.dLogitsReal)), tf.log(1.-self.dLogitsReal)),
        #                              name='realLoss')

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

        # self.pGen = -tf.reduce_mean(tf.multiply(tf.zeros_like(self.dLogitsFake), tf.log(self.dLogitsFake))+
        #                             tf.multiply((1.-tf.zeros_like(self.dLogitsFake)), tf.log(1.-self.dLogitsFake)),
        #                             name='fakeLoss')

        self.squareLoss = tf.reduce_mean(
            tf.square(tf.reshape(self.realData, [-1]) - tf.reshape(self.fakeData, [-1])))  # 文中公式(4)

        self.directLoss = tf.reduce_mean(tf.abs(
            tf.sign(self.fakeData[:, -1, :] - self.realData[:, -2, :]) -            # 文中公式(5)
            tf.sign(self.realData[:, -1, :] - self.realData[:, -2, :])))            # 这一段代码好好检查一下逻辑有没有错误

        self.dLoss = self.pReal + self.pGen
        # self.dLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitsReal,
        #                                                                     labels=tf.ones_like(self.dLogitsReal)) +
        #                             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitsFake,
        #                                                                     labels=tf.zeros_like(self.dLogitsFake)),
        #                                                                     name='dLoss')

        self.gLoss_trickLogD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dLogitsFake, labels=tf.ones_like(self.dLogitsFake)))  # + self.squareLoss + self.directLoss
        self.gLoss_minimax = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                             logits=self.dLogitsFake,
                                             labels=tf.zeros_like(self.dLogitsFake)))  # + self.squareLoss + self.directLoss

        self.gLoss_ls = tf.reduce_mean(tf.square(self.dLogitsFake-1.))   #  + self.squareLoss + self.directLoss

        # self.gLoss_trickLogD = -tf.reduce_mean(tf.multiply(tf.ones_like(self.dLogitsFake), tf.log(self.dLogitsFake)))
        # self.gLoss_minimax = tf.reduce_mean(tf.multiply(1., tf.log(1.-self.dLogitsFake)))


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
        self.FD = tf.gradients(ys=self.dLoss, xs=self.DVars)
        self.grad_sum = [tf.reduce_sum(tf.square(x))for x in self.FD]
        self.Fd_score = self.beta * tf.log(tf.reduce_sum(self.grad_sum))

    def trainAndPredict(self, data, dataProcessor, day):
        """
        传入的需要参与到训练的数据,使得模型进行训练
        :param data:
        :param dataProcessor:
        :return:
        """
        # print(np.array(data))

        """
        data : 为正则化后的数据
        indicators 去除最后一列之后为Generator的输入
        realData 为真实数据，作为Dis的一个输入，是data 的最后一列
        """

        # print(data[-25:, :])
        # print(data.shape)
        batchIndex, indicators, realData = dataProcessor.getTrainData(data)
        batchLen = len(batchIndex)-1
        genInputs = np.array(indicators)[:, :-1, :]  # batch
        testInput = np.array(indicators)[np.newaxis, -1, 1:, :]   # batch 种的最后一个timestep
        # print(testInput)
        # print(testInput.shape)

        trainHist = {}
        trainHist['DLoss'] = []
        trainHist['GLoss'] = []
        trainHist['pReal'] = []
        trainHist['pFake'] = []

        gen_new_params = []
        gen_tem_param = []
        population_param = []
        GLosses, DLosses, pReal, pFake = [], [], [], []
        n_updates = 0

        with tf.Session(graph=self.graph, config=self.my_config).as_default() as self.sess:
            with self.graph.as_default():

                # 初始化参数
                for i in range(1, 11):
                    self.sess.run(tf.global_variables_initializer())
                    Gvar_value = self.sess.run(self.GVars)
                    gen_tem_param.append(Gvar_value)
                    if i % 10 == 0:
                        gen_new_params.append(np.mean(gen_tem_param, axis=0))

                # get all step_index
                step_index, step_index_len = self.get_all_step_index(batchLen)
                epoch = 1

                for st_index in range(step_index_len):
                    print('day %d Epoch %d of %d' % (day, epoch, self.epochs))
                    genIn, real = [], []
                    for i in range(len(step_index[st_index])):
                        step = step_index[st_index][i]
                        genIn.append(genInputs[batchIndex[step]: batchIndex[step + 1]])
                        real.append(np.array(realData[batchIndex[step]: batchIndex[step + 1]]))
                    # 在batchIndex[step]: batchIndex[step +1] 处理成多个
                    if st_index == 0:
                        self.sess.run([self.Gweight_value[0],
                                       self.Gweight_value[1],
                                       self.Gweight_value[2],
                                       self.Gweight_value[3]],
                                      feed_dict={self.Gweight[0]: gen_new_params[0][0],
                                                 self.Gweight[1]: gen_new_params[0][1],
                                                 self.Gweight[2]: gen_new_params[0][2],
                                                 self.Gweight[3]: gen_new_params[0][3],
                                                 })
                        for can_i in range(0, self.ncandi):
                            for step in range(len(step_index[st_index])):
                                _, gLossVal, pFakeVal = self.sess.run([self.gOptim_trickLogD,
                                                                       self.gLoss_trickLogD,
                                                                       self.pGen],
                                                                      feed_dict={
                                                                                self.genInput: genIn[step],
                                                                                self.realData: real[step]
                                                                                })
                            gen_new_params[can_i] = self.sess.run(self.GVars)

                    else:
                        gen_old_params = gen_new_params
                        for can_i in range(0, self.ncandi):
                            frScore = []
                            for type_i in range(self.nloss):
                                self.sess.run([self.Gweight_value[0],
                                               self.Gweight_value[1],
                                               self.Gweight_value[2],
                                               self.Gweight_value[3]],
                                              feed_dict={self.Gweight[0]: gen_old_params[can_i][0],
                                                         self.Gweight[1]: gen_old_params[can_i][1],
                                                         self.Gweight[2]: gen_old_params[can_i][2],
                                                         self.Gweight[3]: gen_old_params[can_i][3]})

                                if self.loss[type_i] == 'trickLogD':
                                    _, gLossVal = self.sess.run([self.gOptim_trickLogD, self.gLoss_trickLogD],
                                                                feed_dict={self.genInput: genIn[type_i],
                                                                           self.realData: real[type_i]})
                                elif self.loss[type_i] == 'minimax':
                                    _, gLossVal = self.sess.run([self.gOptim_minimax, self.gLoss_minimax],
                                                                feed_dict={self.genInput: genIn[type_i],
                                                                           self.realData: real[type_i]})
                                elif self.loss[type_i] == 'ls':
                                    _, gLossVal = self.sess.run([self.gOptim_ls, self.gLoss_ls],
                                                                feed_dict={self.genInput: genIn[type_i],
                                                                           self.realData: real[type_i]})
                                # 计算适应度函数值
                                fr_score, fd_score, dlossvalue, pFakeVal, gradSum,  = self.sess.run(
                                                  [self.fakeOut, self.Fd_score, self.dLoss, self.pGen, self.grad_sum, ],
                                                  feed_dict={
                                                     self.genInput: genIn[type_i],
                                                     self.realData: real[type_i]
                                                  })
                                frScore.append(fr_score)
                                # print(gradSum[0:6])
                                # print(gradSum[6:12])
                                # print(gradSum[12:18])
                                # print(len(gradSum))
                                print("gloss: %r ,Fq : %r, fd : %r, dloss : %r" % (gLossVal, fr_score, fd_score, dlossvalue))

                                fit = fr_score - fd_score

                                if can_i * self.nloss + type_i < self.ncandi:
                                    idx = can_i * self.nloss + type_i
                                    fitness[idx] = fit
                                    fake_rate[idx] = fr_score
                                    gen_new_params[idx] = self.sess.run(self.GVars)
                                else:
                                    fit_com = fitness - fit
                                    if min(fit_com) < 0:
                                        ids_replace = np.where(fit_com == min(fit_com))
                                        idr = ids_replace[0][0]
                                        fitness[idr] = fit
                                        fake_rate[idr] = fr_score
                                        gen_new_params[idr] = self.sess.run(self.GVars)

                            if frScore[0] < 5*1e-5 and frScore[1] < 5*1e-5 and frScore[2] < 5*1e-5:
                                return _, True

                        print("epoch: %d,fake_rate:%r,fitness:%r" % (epoch, fake_rate, fitness))

                    # train Discriminator
                    for step in range(len(step_index[st_index])):
                        _, dLossVal, pRealVal = self.sess.run([self.dOptim, self.dLoss, self.pReal],
                                     feed_dict={
                                         self.genInput: genIn[step],
                                         self.realData: real[step]
                                     })

                    for i in range(0, self.ncandi):
                        tr, fr, fd = self.sess.run([self.realOut, self.fakeOut, self.Fd_score],
                                                   feed_dict={
                                                       self.genInput: genIn[0],
                                                       self.realData: real[0]
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

                    print("real_rate: %f, fake_rate : %f, FDL :%f" % (real_rate, fake_rate, FDL))

                    # sess.run(self.clipD)

                    GLosses.append(gLossVal)
                    DLosses.append(dLossVal)
                    pReal.append(pRealVal)
                    pFake.append(pFakeVal)

                    for _ in range(len(step_index[st_index])):
                        n_updates += 1
                        if (n_updates % (batchLen) == 0):
                            epoch += 1
                            trainHist['DLoss'].append(np.mean(DLosses))
                            trainHist['GLoss'].append(np.mean(GLosses))
                            trainHist['pReal'].append(np.mean(pReal))
                            trainHist['pFake'].append(np.mean(pFake))
                            GLosses, DLosses, pReal, pFake = [], [], [], []


            if not os.path.isdir('./loss_bilstm_GAN/'):
                os.mkdir('./loss_bilstm_GAN/')

            if not os.path.isdir('./loss_bilstm_GAN/%d/' % day):
                os.mkdir('./loss_bilstm_GAN/%d/' % day)

            dataframe = pd.DataFrame({'real': trainHist['pReal'], 'fake': trainHist['pFake']})
            dataframe.to_csv('./loss_bilstm_GAN/%d/realFakeLoss.csv' % day, index=False, sep=',')

            dataframe = pd.DataFrame({'D': trainHist['DLoss'], 'G': trainHist['GLoss']})
            dataframe.to_csv('./loss_bilstm_GAN/%d/GANLoss.csv' % day, index=False, sep=',')

            # 至此模型的训练以及相关数据的保存已经完毕，接下来预测后一天的价格
            price = self.sess.run(self.predictValue,
                                  feed_dict={self.genInput: testInput})

        return price, False


if __name__ == '__main__':
    para = dict(timeStep=21, hiddenUnit=32, GeneratorInputSize=13, GeneratorOutputSize=1,
                 discriminatorInputSize=1, dim=16, c=0.01, learningRate=0.00001, epochs=5, outputGraph=True)
    model = GANFD(**para)
    model.trainAndPredict()
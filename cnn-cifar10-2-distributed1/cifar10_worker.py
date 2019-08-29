# coding:utf-8
import tensorflow as tf
import numpy as np
import cifar10_input
import cifar10_model
import matplotlib.pyplot as plt
import os
import time
import random
from filelock import FileLock


"""
添加参数
"""
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('model_name', 'worker1', 'worker1 or worker2')
# tf.app.flags.DEFINE_string('weight_loss_file', 'weight1_loss', 'weight1_loss.npz or weight2_loss.npz ....')
tf.app.flags.DEFINE_string('GPU_index', '0', ' -1.2.1.2 is disable , if the server has 8 GPUS, 0 ~ 7 '
                                             'represent a index of the GPU to use .')
tf.app.flags.DEFINE_boolean('worker_flag', False, 'specify the value of current worker flag'
                                                  'e.g. False or True')
tf.app.flags.DEFINE_integer('worker_index', 1, 'specify the index of the current worker'
                                               'e.g. one of the range(worker_nums)')
tf.app.flags.DEFINE_integer('worker_nums', 1, 'specify the number of the workers')

# 指定不需要GUI的backend
plt.switch_backend('agg')
# 使用GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_index

EPOCHES = 201
DATA_DIR = cifar10_model.DATA_DIR
BATCH_SIZE = cifar10_model.BATCH_SIZE
save_path = './signal_GPU/saver{0}'.format(FLAGS.worker_index)
picture_path = './signal_GPU/results'


# 写flag
def write_flag(x):
    if os.path.exists('flag.txt'):
        if os.path.getsize('flag.txt'):
            flag = np.loadtxt('flag.txt')
            if flag.size != 1:
                if [i for i in range(len(flag)) if flag[i] != 1] == [] and len(flag) == FLAGS.worker_nums:
                    open('flag.txt', 'w').truncate()

    with open('flag.txt', 'a') as f:
        f.write('%d ' % x)
        f.flush()
        f.close()


def read_flag(train_variable):
    """
    写flag
    :param train_variable:将训练变量作为参数传进来，方便最后将更新后的值重新赋值给变量
    :return:
    """

    flag = np.loadtxt('flag.txt')
    if flag.size != 1:
        if [i for i in range(len(flag.size)) if flag[i] != 1] == [] and len(flag.size) == FLAGS.worker_nums:
            FLAGS.worker_flag = False
            weight_loss = np.load("weight{0}_loss.npz".format(FLAGS.worker_index))
            for i in range(len(train_variable)):
                train_variable[i].assign(weight_loss['arr_0'][i])
        else:
            time.sleep(random.random())  # random.random()随机生成一个0~1之间的数


# 训练
def train():
    # 读取图片并带入网络计算
    images, labels = cifar10_input.distorted_inputs(DATA_DIR, BATCH_SIZE)
    t_logits = cifar10_model.inference(images, 'worker{0}'.format(FLAGS.worker_index))
    # 损失值
    t_loss = cifar10_model.loss(t_logits, labels)
    tf.summary.scalar('loss_value', t_loss)
    # 优化器
    global_step = tf.Variable(0, trainable=False)
    t_optimizer = cifar10_model.train_step(t_loss, global_step)
    # 准确值
    t_accuracy = cifar10_model.accuracy(t_logits, labels) # 训练集正确率计算
    tf.summary.scalar('accuracy_value', t_accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    train_variable = tf.trainable_variables()
    Accuracy_value = []
    Loss_value = []
    # 设定定量的GPU显存使用量（取消）
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter('signal_GPU/logs', sess.graph)
        # 记录训练开始时间
        start_time = time.time()
        # 用于记录每一千步训练花费时间
        duration_start_time = time.time()
        for index in range(EPOCHES):
            _, loss_value, accuracy_value, summary = sess.run([t_optimizer, t_loss, t_accuracy, merged])
            Accuracy_value.append(accuracy_value)
            Loss_value.append(loss_value)
            if index % 10 == 0:
                # 用于记录每一千步训练花费时间
                duration_end_time = time.time()
                duration_time = duration_end_time - duration_start_time
                print('index:', index, 'Time:', duration_time, ' loss_value:', loss_value, ' accuracy_value:', accuracy_value)
                duration_start_time = duration_end_time
            train_writer.add_summary(summary, index)
            if index > 0 and index % 100 == 0:
                # 将参数的值全部取出

                train_variable_value = sess.run(train_variable)

                """
                # 将参数的值保存到文件中，以方便ps读取
                # FLAGS.weight_loss_file 保存的文件
                # train_variable_value 保存的值
                # %e 以科学记数法的格式进行保存
                """
                np.savez('weight{0}_loss'.format(FLAGS.worker_index), train_variable_value, [loss_value])
                write_flag(int(0))
                FLAGS.worker_flag = True
                while FLAGS.worker_flag:
                    read_flag(train_variable)

        # 记录训练结束时间
        end_time = time.time()
        training_time = end_time-start_time
        print('Training Time is %f' % training_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(sess, os.path.join(save_path, 'model.ckpt'))

        # 检查保存图片的文件夹是否存在
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)
        # accuracy value（取消accuracy画图）
        plt.figure(figsize=(20, 10))
        plt.plot(range(EPOCHES), Accuracy_value)
        plt.xlabel('training step')
        plt.ylabel('accuracy value')
        plt.title('the accuracy value of training data')
        plt.savefig('signal_GPU/results/accuracy{0}.png'.format(FLAGS.worker_index))

        # loss1 value（取消loss画图）
        plt.figure()
        plt.plot(range(EPOCHES), Loss_value)
        plt.xlabel('training value')
        plt.ylabel('loss1 value')
        plt.title('the value of the loss1 function of the training data')
        plt.savefig('signal_GPU/results/loss1{0}.png'.format(FLAGS.worker_index))

        #
        train_writer.close()
        coord.request_stop()
        coord.join(threads)


# 验证
def evaluation():
    with tf.Graph().as_default():
        n_test = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        eval_images, eval_lables = cifar10_input.inputs(DATA_DIR, BATCH_SIZE)
        eval_logits = cifar10_model.inference(eval_images, 'worker{0}'.format(FLAGS.worker_index))
        # tf.nn.in_top_k(predictions, targets, k, name=None)
        # 每个样本的预测结果的前k个最大的数里面是否包括包含targets预测中的标签，一般取1，
        # 即取预测最大概率的索引与标签的对比
        top_k_op = tf.nn.in_top_k(eval_logits, eval_lables, 1)
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('signal_GPU/saver{0}'.format(FLAGS.worker_index))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            num_iter = int(n_test / BATCH_SIZE)
            true_count = 0
            for step in range(num_iter):
                predictions = session.run(top_k_op)
                true_count = true_count + np.sum(predictions)
            precision = true_count / (num_iter * BATCH_SIZE)
            print('precision=', precision)
            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    train()
    evaluation()


if __name__ == '__main__':
    tf.app.run()

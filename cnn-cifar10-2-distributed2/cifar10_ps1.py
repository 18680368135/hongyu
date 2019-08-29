import tensorflow as tf
import numpy as np
import time
import random
from scipy._lib._util import check_random_state
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('GPU_index', '0', ' -1.2.1.2 is disable , if the server has 8 GPUS, 0 ~ 7 '
                                             'represent a index of the GPU to use .')
tf.app.flags.DEFINE_integer('worker_nums', 1, 'specify the number of the workers')
tf.app.flags.DEFINE_integer('max_population', 8, 'the max num of the population')

# 使用GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_index

# 定义伸缩因子
scale = 0.5


def bulid_init_population(n):
    """
    获得初始种群
    :param n:代表种群的个数
    :return: 返回n个种群的参数值和损失值
    """
    weight_loss = []
    for i in range(n):
        weight_loss.append(np.load('weight{0}_loss.npz'.format(i)))
    print(np.shape(weight_loss), len(weight_loss))
    return weight_loss


"""
产生一个二位数组，用于存放weight_loss中的loss值，以及loss值得原有顺序，便于后面的排序
"""


def arr_def(MyList):
    a = [[], []]
    for m in range(2):
        for n in range(len(MyList)):
            if m == 0:
                a[m].append(n)
            else:
                a[m].append(MyList[n]['arr_1'][0])
    return a


def QuickSort(a, start, end):

    # 判断low是否小于high,如果为false,直接返回
    if start < end:
        i, j = start, end
        # 设置基准数
        base = a[1][i]
        base_1 = a[0][i]

        while i < j:
            # 如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while (i < j) and (a[1][j] >= base):
                j = j - 1

            # 如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
            a[0][i] = a[0][j]
            a[1][i] = a[1][j]

            # 同样的方式比较前半区
            while (i < j) and (a[1][i] <= base):
                i = i + 1
            a[0][j] = a[0][i]
            a[1][j] = a[1][i]

        # 做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
        a[1][i] = base
        a[0][i] = base_1
        # 递归前后半区
        QuickSort(a, start, i - 1)
        QuickSort(a, j + 1, end)
    # print("loss1 is {0}".format(a[1.2.1.2]))
    return a


"""
扩充种群过程中将新得到得种群个体插入到当前种群并排序
"""


def insert_sort(weight_loss_sort, max_num, current_population):

    for i in range(len(weight_loss_sort)):
        for j in range(len(current_population)):
            if current_population[j]['arr_1'][0] >= weight_loss_sort[i]['arr_1'][0]:
                current_population.insert(j, weight_loss_sort[i])
                break
            elif current_population[len(current_population) - 1]['arr_1'][0] < weight_loss_sort[i]['arr_1'][0]:
                current_population.append(weight_loss_sort[i])
                break
        if len(current_population) > max_num:
            current_population.pop()
    return current_population


def base_population_sort(weight_loss):

    weight_loss_sort = []
    base_arr = QuickSort(arr_def(weight_loss), 0, len(weight_loss) - 1)
    print("loss1 is {0}".format(base_arr[1]))
    for i in range(len(base_arr[0])):
        weight_loss_sort.append(weight_loss[base_arr[0][i]])

    return weight_loss_sort


def update_base_population_sort(weight, max_num, weight_loss):

    weight_loss_sort = []
    base_arr = QuickSort(arr_def(weight_loss), 0, len(weight_loss) - 1)
    print("loss1 is {0}".format(base_arr[1]))
    for i in range(len(base_arr[0])):
        weight_loss_sort.append(weight_loss[base_arr[0][i]])
    weight = insert_sort(weight_loss_sort, max_num, weight)
    return weight


"""
扩充种群中的个体，直到种群个体数达到最大值，停止扩充，但是每次获得新个体后都要保证当前种群最优。
"""


def expand_population(max_num, weight_loss, current_population):

    if len(current_population) == 0:
        base_arr = QuickSort(arr_def(weight_loss), 0, len(weight_loss) - 1)
        for i in range(len(base_arr[0])):
            current_population.append(weight_loss[base_arr[0][i]])
    else:
        weight_loss_sort = []
        base_arr = QuickSort(arr_def(weight_loss), 0, len(weight_loss) - 1)
        for i in range(len(base_arr[0])):
            weight_loss_sort.append(weight_loss[base_arr[0][i]])
        return insert_sort(weight_loss_sort, max_num, current_population)
    return current_population


def best1(population, samples):
    """
    best1bin, best1exp
    """
    r0, r1 = samples[:2]
    return (population[0]['arr_0'] + scale *
            (population[r0]['arr_0'] - population[r1]['arr_0']))


def rand1(population, samples):
    """
    rand1bin, rand1exp
    """
    r0, r1, r2 = samples[:3]
    return (population[r0]['arr_0'] + scale *
            (population[r1]['arr_0'] - population[r2]['arr_0']))


def randtobest1(current_population, samples):
    """
    randtobest1bin, randtobest1exp
    """
    r0, r1, r2 = samples[:3]
    bprime = np.copy(current_population[r0]['arr_0'])
    bprime += scale * (current_population[0]['arr_0'] - bprime)
    bprime += scale * (current_population[r1]['arr_0'] -
                       current_population[r2]['arr_0'])
    return bprime


def currenttobest1(population, current_population, candidate, samples):
    """
    currenttobest1bin, currenttobest1exp
    """
    r0, r1 = samples[:2]
    bprime = (population[candidate]['arr_0'] + scale *
              (current_population[0]['arr_0'] - population[candidate]['arr_0']) +
              scale * (current_population[r0]['arr_0'] - current_population[r1]['arr_0']))
    return bprime


def best2(current_population, samples):
    """
    best2bin, best2exp
    """
    r0, r1, r2, r3 = samples[:4]
    bprime = (current_population[0]['arr_0'] + scale *
              (current_population[r0]['arr_0'] + current_population[r1]['arr_0'] -
               current_population[r2]['arr_0'] - current_population[r3]['arr_0']))

    return bprime


def rand2(population, samples):
    """
    rand2bin, rand2exp
    """
    r0, r1, r2, r3, r4 = samples
    bprime = (population[r0]['arr_0'] + scale *
              (population[r1]['arr_0'] + population[r2]['arr_0'] -
               population[r3]['arr_0'] - population[r4]['arr_0']))

    return bprime


def select_samples(candidate, number_samples, base_population):
    """
    obtain random integers from range(len(current_population)),
    without replacement.  You can't have the original candidate either.
    """
    idex = list(range(len(base_population)))
    idex.remove(candidate)
    check_random_state(None).shuffle(idex)
    idex = idex[:number_samples]

    return idex


def mutation_to_save(mutation, x):
    """
    将变异后的个体，与厨师种群个体一一对应
    :param mutation:
    :return:
    """
    with open('flag.txt', 'a') as f:
        for i in range(len(mutation)):
            np.savez('weight{0}_loss'.format(i), mutation[i])
            f.write('%d' % x)
        f.flush()
        f.close()


def main():
    init_population_num = FLAGS.worker_nums
    weight = []
    m = int(0)
    while True:
        if os.path.exists('flag.txt'):
            # print("flag 文件存在")
            if os.path.getsize('flag.txt'):
                # print("flag 文件不为空")
                flag = open('flag.txt').read()
                # flag = np.loadtxt('flag.txt')
                if [i for i in range(len(flag)) if int(flag[i]) != 0] == [] and len(flag) == FLAGS.worker_nums:
                    print("第 {0} 次变异开始".format(m))
                    open('flag.txt', 'w').truncate()
                    base_population = bulid_init_population(init_population_num)
                    current_population = base_population_sort(base_population)

                    # current_population = expand_population(FLAGS.max_population, base_population, current_population)
                    # print(np.shape(current_population), len(current_population))
                    mutation = []
                    for candidate in range(init_population_num):
                        sample = select_samples(candidate, init_population_num, base_population)
                        bprime = currenttobest1(base_population, current_population, candidate, sample)
                        mutation.append(bprime)
                    mutation_to_save(mutation, 1)

                    m += 1
                else:
                    # print("休息一会")
                    time.sleep(random.random())


if __name__ == "__main__":
    main()

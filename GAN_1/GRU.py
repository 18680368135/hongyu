import tensorflow as tf


class GRU(object):
    def __init__(self, rnn_unit, input_size, output_size, X):
        self.rnn_unit = rnn_unit
        self.input_size = input_size
        self.output_size = output_size

        self.X = X          # X的placeholder

        self._build_net()

    def gru(self, X):
        # print(X)
        batch_size = tf.shape(X)[0]
        # print(batch_size)

        # cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit, forget_bias=1.0, state_is_tuple=True)
        cell = tf.nn.rnn_cell.GRUCell(self.rnn_unit)
        # [cell]*2为2层LSTM
        mgru_cell = tf.contrib.rnn.MultiRNNCell([cell] * 1, state_is_tuple=True)
        init_state = mgru_cell.zero_state(batch_size, dtype=tf.float32)

        output_rnn, final_states = tf.nn.dynamic_rnn(mgru_cell, X, initial_state=init_state, dtype=tf.float32)
        # print(output_rnn)
        # output = tf.reshape(output_rnn, [-1.2.1.2, self.rnn_unit])
        # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
        # print(output)
        pred = tf.layers.dense(output_rnn, self.output_size, name='output')

        # print(pred)
        return pred, final_states

    def _build_net(self):
        self.pred, _ = self.gru(self.X)


if __name__ == '__main__':
    prediction = tf.placeholder(tf.float32, [None, 20, 1], name='prediction_input')
    # input = {'rnn_unit': 10, 'input_size': 1.2.1.2, 'output_size': 3, 'X':prediction}
    input = dict(rnn_unit=21, input_size=1, output_size=1, X=prediction)
    # lstm = LSTM(input)
    gru = GRU(**input)


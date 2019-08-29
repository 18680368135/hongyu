import tensorflow as tf


class BiLSTM(object):
    def __init__(self, rnn_unit, input_size, output_size, X):
        self.rnn_unit = rnn_unit
        self.input_size = input_size
        self.output_size = output_size

        self.X = X          # X的placeholder

        self._build_net()

    def bilstm(self, X):
        # print(X) [32 20 1]
        # batch_size = tf.shape(X)[0]
        # seq_length = tf.shape(X)[0]
        # print(batch_size)

        mlstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit, forget_bias=1.0, state_is_tuple=True)
        mlstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit, forget_bias=1.0, state_is_tuple=True)
        # [cell]*2为2层LSTM
        # mlstm_cell = tf.contrib.rnn.MultiRNNCell([cell] * 1, state_is_tuple=True)
        # mlstm_cell_bw = tf.contrib.rnn.MultiRNNCell([cell] * 1, state_is_tuple=True)

        # init_state_fw = mlstm_cell.zero_state([batch_size, seq_length], dtype=tf.float32)
        # init_state_bw = mlstm_cell_bw.zero_state([batch_size, seq_length], dtype=tf.float32)

        # output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, X, initial_state=init_state, dtype=tf.float32)
        output_rnn, final_states = tf.nn.bidirectional_dynamic_rnn(mlstm_cell, mlstm_cell_bw, X,
                                                                   dtype=tf.float32)
        # print(output_rnn)
        output = tf.concat(output_rnn, 2)
        # print(output)

        # output = tf.transpose(output, [1, 0, 2])
        # fw_output_rnn, _ = output_rnn
        out_state = tf.concat(final_states, axis=-1)

        # print(output_rnn)
        # output = tf.reshape(output_rnn, [-1.2.1.2, self.rnn_unit])
        # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
        # print(output)
        # pred = tf.layers.dense(output, self.output_size, name='output')
        pred = tf.layers.dense(output, self.output_size, name='output')

        # print(pred[:,-1,:])
        return pred, out_state

    def _build_net(self):
        self.pred, _ = self.bilstm(self.X)


if __name__ == '__main__':
    prediction = tf.placeholder(tf.float32, [None, 10, 1], name='prediction_input')
    # input = {'rnn_unit': 10, 'input_size': 1.2.1.2, 'output_size': 3, 'X':prediction}
    input = dict(rnn_unit=10, input_size=1, output_size=3, X=prediction)
    # lstm = LSTM(input)
    lstm = BiLSTM(**input)


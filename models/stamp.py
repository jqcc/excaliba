import tensorflow as tf

from base.NN import NeuralNetWork


class Model(NeuralNetWork):
    def __init__(self, train_data, test_data, embedding, n_nodes, model_name, data_name,
                 epoch=30, batch_size=100, embedding_size=100, lr=0.001, lr_dc=0.5, lr_dc_step=3):
        super().__init__(train_data, test_data, embedding, n_nodes, model_name, data_name,
                         epoch, batch_size, embedding_size, lr, lr_dc, lr_dc_step)

        self.last_click = tf.placeholder(tf.int32, [None], name='last')

    def build_net(self):
        # 将填充值0对应的embedding向量设置为0
        self.embedding[0] *= 0
        self.embedding_ = tf.Variable(self.embedding, trainable=self.is_train, dtype=tf.float32)

        inputs = tf.nn.embedding_lookup(self.embedding_, self.inputs_)
        last = tf.nn.embedding_lookup(self.embedding_, self.last_click)

        # 输入的平均池化
        # ([b,n,d]->[b,d]) / ([b,n]->[b,1])
        pool_out = tf.div(tf.reduce_sum(inputs, 1), tf.cast(self.seq_len, tf.float32))

        w1 = tf.Variable(tf.random_normal([self.batch_size, self.embedding_size, self.embedding_size], stddev=0.05), trainable=True)
        w2 = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=0.05), trainable=True)
        w3 = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=0.05), trainable=True)
        w4 = tf.Variable(tf.random_normal([self.embedding_size, 1], stddev=0.05), trainable=True)
        b1 = tf.Variable(tf.random_normal([self.embedding_size], stddev=0.05), trainable=True)

        # Attention Net
        # [b, n, d]
        inputs_h = tf.matmul(inputs, w1)
        # [b, 1, d]
        last_h = tf.reshape(tf.matmul(last, w2), [-1, 1, self.embedding_size])
        pool_h = tf.reshape(tf.matmul(pool_out, w3), [-1, 1, self.embedding_size])
        m = tf.nn.sigmoid(inputs_h + last_h + pool_h + b1)
        alpha = tf.matmul(tf.reshape(m, [-1, self.embedding_size]), w4) * tf.reshape(self.mask, [-1, 1])
        # TODO 看看这个b
        att_out = tf.reduce_sum(tf.reshape(alpha, [self.batch_size, -1, 1]) * inputs, 1)

        # ========== MLP Cell A & B
        w5 = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=0.05), trainable=True)
        w6 = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=0.05), trainable=True)

        cell_a = tf.tanh(tf.matmul(att_out, w5))
        cell_b = tf.tanh(tf.matmul(last, w6))
        prod = cell_a * cell_b

        candidates = self.embedding_[1:]
        # [b, d] x [n, d].T
        out = tf.matmul(prod, candidates, transpose_b=True)
        self.logits = out

    def get_feed_dict(self, batch, keep_prob):
        sessions, masks, labels = batch[0], batch[1], batch[2]
        last = []
        for session, mask in zip(sessions, masks):
            last.append(session[sum(mask)-1])

        self.feed = {
            self.inputs_: sessions,
            self.labels_: labels,
            self.mask: masks,
            self.seq_len: list(map(sum, masks)),
            self.keep_prob_: keep_prob,
            self.last_click: last
        }


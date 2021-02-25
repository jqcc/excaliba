import tensorflow as tf

from base.NN import NeuralNetWork


class Model(NeuralNetWork):
    def __init__(self, train_data, test_data, embedding, n_nodes, model_name, data_name,
                 epoch=30, batch_size=100, embedding_size=100, lr=0.001, lr_dc=0.5, lr_dc_step=3):
        super().__init__(train_data, test_data, embedding, n_nodes, model_name, data_name,
                         epoch, batch_size, embedding_size, lr, lr_dc, lr_dc_step)

        self.last_click = tf.placeholder(tf.int32, [None], name='last')

    def build_net(self):
        self.embedding_ = tf.Variable(self.embedding, trainable=self.is_train)
        last = tf.nn.embedding_lookup(self.embedding_, self.last_click)
        candidates = self.embedding_[1:]
        # [b, d] x [n, d].T
        out = tf.matmul(last, candidates, transpose_b=True)
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


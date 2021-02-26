from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np
import os


class NeuralNetWork(metaclass=ABCMeta):
    def __init__(self, train_data, test_data, embedding, n_nodes, model_name, data_name,
                 epoch=30, batch_size=100, embedding_size=100, lr=0.001, lr_dc=0.5, lr_dc_step=3):
        self.train_data = train_data
        self.test_data = test_data
        self.embedding = embedding
        self.n_nodes = n_nodes
        self.epoch = epoch
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.learning_rate = lr
        self.learning_rate_decay = lr_dc
        self.decay_steps = lr_dc_step * len(train_data.inputs) / batch_size
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.model_name = model_name
        self.data_name = data_name
        # 注意模型保存 是按算法为大类，数据为小类
        # 因为模型训练结果，只有本模型能加载
        self.model_save_path = "checkpoints/" + model_name + "/" + data_name
        self.model_save_file = self.model_save_path + "/" + data_name + ".ckpt"
        # embeddings保存 是按数据为大类，算法为小类
        # 因为embeddings与具体模型关联不大，一个模型训练的embeddings通常会给别的模型用
        self.embed_save_path = "embeddings/" + data_name
        self.embed_save_file = self.embed_save_path + "/" + model_name + ".npy"

        self.inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.labels_ = tf.placeholder(tf.int32, [None], name='labels')
        self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
        self.seq_len = tf.placeholder(tf.int32, [None], name='lens')
        self.keep_prob_ = tf.placeholder(tf.float32, name='keep')
        self.feed = None
        self.is_train = True
        self.embedding_ = None

        self.logits = None
        self.loss = None
        self.opt = None

    @abstractmethod
    def build_net(self):
        """
        模型子类在此函数内实现模型
        input: self.inputs
        :return: 方法内设置self.logits
        """
        pass

    def build(self):
        self.build_net()
        # 注意对于label，结果集中是不应该包含填充值0的 因此此处做了-1操作
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_-1, logits=self.logits))
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
                                                        decay_steps=self.decay_steps,
                                                        decay_rate=self.learning_rate_decay, staircase=False)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    @abstractmethod
    def get_feed_dict(self, batch, keep_prob):
        """读取一个batch，组织placeholder"""
        pass

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 0
            self.is_train = True

            loss_ = []
            for e in range(self.epoch):
                print('epoch: ', e, '===========================================')
                slices = self.train_data.generate_batch(self.batch_size)
                fetches = [self.opt, self.loss, self.global_step]

                for i in slices:
                    self.get_feed_dict(self.train_data.get_slice(i), 0.5)
                    _, loss, _ = sess.run(fetches, feed_dict=self.feed)
                    loss_.append(loss)

                    if iteration % 100 == 0:
                        print("Epoch: {}/{}".format(e+1, self.epoch),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(np.mean(loss_)))
                    iteration += 1
                print("Epoch: {}/{}".format(e + 1, self.epoch),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(np.mean(loss_)))

                self.save_model(sess)
                self.test()

    def test(self):
        best_result = [0, 0]
        saver = tf.train.Saver()
        self.is_train = False

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            slices = self.test_data.generate_batch(self.batch_size)
            fetchs = [self.loss, self.logits]

            hit, mrr, test_loss_ = [], [], []
            for i in slices:
                batch = self.test_data.get_slice(i)
                labels = batch[2]
                self.get_feed_dict(batch, 1.0)
                loss, scores = sess.run(fetchs, feed_dict=self.feed)

                test_loss_.append(loss)
                index = np.argsort(scores, 1)[:, -20:]
                for score, target in zip(index, labels):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))
            hit = np.mean(hit) * 100
            mrr = np.mean(mrr) * 100
            test_loss = np.mean(test_loss_)
            if hit >= best_result[0]:
                best_result[0] = hit

            if mrr >= best_result[1]:
                best_result[1] = mrr
            print('test_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (test_loss, best_result[0], best_result[1]))

    def save_model(self, sess):
        saver = tf.train.Saver()
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        saver.save(sess, self.model_save_file)
        embeddings = sess.run(self.embedding_)
        if not os.path.exists(self.embed_save_path):
            os.makedirs(self.embed_save_path)
        np.save(self.embed_save_file, embeddings)

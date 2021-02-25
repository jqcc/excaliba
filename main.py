import argparse
from importlib import import_module

from loader import Data, count_nodes, gen_embedding, load_embedding

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='sample', help='dataset name')
parser.add_argument('-m', '--model', type=str, default='sample', help='model name')
parser.add_argument('-e', '--epoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('-b', '--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('-t', '--test', action='store_true', help='train or test')
parser.add_argument('--embedding', type=str, default='random', help='random embedding or load trained embedding')
parser.add_argument('--embedding_size', type=int, default=100, help='embedding size')

opt = parser.parse_args()

# load data
TRAIN_PATH = "datas/" + opt.dataset + "/train_.txt"
TEST_PATH = "datas/" + opt.dataset + "/test_.txt"
train_data = Data(TRAIN_PATH)
test_data = Data(TEST_PATH)

print("data loaded")

n_nodes = count_nodes(train_data, test_data)
print("node numbers: ", n_nodes)

# gen embedding or load embedding
if opt.embedding == "random":
    embedding = gen_embedding(n_nodes, opt.embedding_size)
else:
    embedding = load_embedding(opt.dataset, opt.embedding)
print("embedding loaded")

x = import_module('models.' + opt.model)
model = x.Model(train_data, test_data, embedding, n_nodes, opt.model, opt.dataset,
                opt.epoch, opt.batchSize, opt.embedding_size, opt.lr, opt.lr_dc, opt.lr_dc_step)
model.build()
# train
if not opt.test:
    model.train()
    model.test()

# test
else:
    model.test()

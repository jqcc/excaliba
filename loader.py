import numpy as np


def open_file(filename):
    """将文件加载到内存中，以list的方式存储原始数据"""
    with open(filename, "r") as f:
        datas = []
        for line in f.readlines():
            # 以空格切分一行 并转换为int后 存储在data中
            datas.append(list(map(int, line.split(" "))))

        return datas


def split_data_and_label(datas):
    """切分出输入数据与标签数据 一条数据的最后一条未标签 其余为输入数据"""
    data_with_label = [[], []]
    for d in datas:
        data_with_label[0].append(d[:-1])
        data_with_label[1].append(d[-1])
    return data_with_label


def data_masks(all_usr_pois):
    """计算数据掩码"""
    # 使用0做输入的padding
    item_tail = [0]
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data:
    def __init__(self, filename, shuffle=False):
        data = split_data_and_label(open_file(filename))
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        """生成batch"""
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        """取一个batch
            此处仅返回单条session，mask，target，不做任何预处理
            预处理放在model内部，方便对每个model自定义其希望的输入格式
        """
        return self.inputs[index], self.mask[index], self.targets[index]


def count_nodes(*datas):
    """计算节点数"""
    node2id = set()
    for data in datas:
        for inp, tar in zip(data.inputs, data.targets):
            for n in inp:
                if n not in node2id:
                    node2id.add(n)
            if tar not in node2id:
                node2id.add(tar)

    return max(node2id)+1


def load_embedding(dataset, model):
    """加载训练好的embedding"""
    return np.load("embeddings/"+dataset+"/"+model+".npy")


def gen_embedding(n_nodes, embed_size):
    """随机初始化embedding矩阵"""
    return np.random.normal(0, 0.05, [n_nodes, embed_size])


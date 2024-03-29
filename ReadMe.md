# 算法说明

## 功能描述

训练时使用gpu加载，使用时使用cpu加载，可以使用如下环境变量
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
```

在`main.py`中使用下面一条环境变量屏蔽了tf中的通知信息，如需打印，将其设置为`0`或注释掉即可。

另外，将其设置为`2`表示`屏蔽通知信息和警告信息`，将其设置为`3`表示`屏蔽通知信息、警告信息和报错信息`。
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
```

## 依赖

* numpy==1.19.2
* tensorflow==1.14

## 训练

### 数据集

数据集统一存放在`datas`目录下，以数据集名称命名目录，如`/datas/sample/train_.txt`。

数据集分: 训练集`(train_.txt)`, 测试集`(test_.txt)`。

数据格式：一条session为一行，其中最后一项为标签，前面若干项为当前会话中的历史点击记录。
```text
1 2 3
1 2
4 5
6 7
```
此处有4条会话数据，对于第一条会话`[1 2 3]`，其中`[1 2]`为历史点击记录，`3`为待预测的下一次点击。

训练时使用`-d dataset`指定要加载的数据集，模型会依据数据名称自动到`datas`目录下加载对应数据

## 使用说明

### 训练模型
将数据集放在`datas`目录下之后，启动模型训练，根据需要设定超参数
```bash
$ python main.py -m model_name -d dataset
$ python main.py -m sample -d sample -e 10
```

### 测试模型
将测试集放在`datas`目录下之后，启动模型测试
```bash
$ python main.py -m model_name -d dataset -t
```

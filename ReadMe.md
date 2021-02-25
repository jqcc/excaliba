# 算法说明

## 功能描述

训练时使用gpu加载，使用时使用cpu加载
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
```

## 依赖

numpy==1.19.2
tensorflow==1.14

## 训练

### 数据集
数据集需要以给定格式，给定名称提供，否则不予识别
数据集需要放在datas目录下，以数据集名称命名目录，训练时使用-d dataset指定要加载的数据集，模型会依据数据名称自动到datas目录下加载对应数据

## 输入说明


## 输入示例
{
    
}

## 输出说明


## 输出示例
{
    
}





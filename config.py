#!usr/bin/env python
# -*- coding:utf-8 -*-

# 数据参数设置
PAD = 0  # padding占位符的索引
# 模型参数设置
N = 6  # transformer中encoder、decoder层数
HEADS = 8  # 多头注意力个数
D_MODEL = 512  # 输入、输出词向量维数
D_FFN = 2048  # feed forward全连接层维数
DROPOUT_PROB = 0.1  # dropout比例
MAX_LENGTH = 60  # 语句最大长度

# 数据集路径设置
TRAIN_FILE = 'nmt/en-cn/train.txt'  # 训练集
DEV_FILE = "nmt/en-cn/train.txt"  # 验证集

SAVE_FILE = 'save'  # 模型保存路径
DEVICE = "cuda"

# 训练参数设置
BATCH_SIZE = 32  # 批次大小
EPOCHS = 20  # 训练轮数
NUM_WORKERS = 0
IS_SHUFFLE = True
DROP_LASR = True  # 如果最后一个batch的数据量小于指定的batch_size，是否丢弃
LR = 10 ** -2
MOMENTUM = 0.99

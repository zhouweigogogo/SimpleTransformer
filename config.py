#!usr/bin/env python
# -*- coding:utf-8 -*-

# 数据参数设置
PAD = 0 

# 模型参数设置
N = 6  # transformer中encoder、decoder层数
HEADS = 8  # 多头注意力个数
D_MODEL = 512  # 输入、输出词向量维数
D_FFN = 2048  # feed forward全连接层维数
DROPOUT_PROB = 0.1  # dropout比例
MAX_LENGTH = 60  # 语句最大长度

# 数据集路径设置
DATA_FILE = 'nmt/en-cn/cmn.txt'  # 数据集

SAVE_PATH = './output/train4'  # 模型保存路径
DEVICE = "cuda"

# 训练参数设置
BATCH_SIZE = 32  # 批次大小
EPOCHS = 30  # 训练轮数
NUM_WORKERS = 0
IS_SHUFFLE = True
DROP_LASR = True  # 如果最后一个batch的数据量小于指定的batch_size，是否丢弃
LR = 10 ** -2
MOMENTUM = 0.99
WARMUP = 4

'''
train1: baseline
train2: batch Adam
train3: batch SGD   1和3的效果感觉差不多，证明batch长度接近只能帮助收敛，不能提升效果

'''
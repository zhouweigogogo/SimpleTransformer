#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
注意：使用中文语料训练Transformer模型时，中文语句一般**以字为单位进行切分**，即无需对中文语句分词。
注意：**同一批次中seq_len相同，不同批次间seq_len可能变化。**
"""

import numpy as np
from nltk import word_tokenize
from collections import Counter
import config
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class MyDatasets(Dataset):
    def __init__(self, data_path, max_seq_len):
        self.max_seq_len = max_seq_len

        self.PAD = torch.tensor([0], dtype=torch.int64)
        self.BOS = torch.tensor([1], dtype=torch.int64)
        self.EOS = torch.tensor([2], dtype=torch.int64)

        # 读取数据、分词
        self.data_src, self.data_tgt = self.load_data(data_path)
        # 构建词表(word2index, word_len, index2word)
        self.src_word_dict, self.src_vocab_size, self.src_index_dict = self.build_src_dict(self.data_src)
        self.tgt_word_dict, self.tgt_vocab_size, self.tgt_index_dict = self.build_tgt_dict(self.data_tgt)

    def load_data(self, path):
        """
        读取英文、中文数据
        对每条样本分词并构建包含起始符和终止符的单词列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你'], ['BOS', '我', '也', '是''], ...]
        """
        en = []
        cn = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split("\t")
                sent_en = sent_en.lower()
                sent_en = word_tokenize(sent_en)
                # 中文按字符切分
                sent_cn = [char for char in sent_cn]
                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_src_dict(self, sentences, max_words=5e4):
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加PAD
        ls = word_count.most_common(int(max_words))
        src_vocab_size = len(ls) + 1
        src_word_dict = {w[0]: index + 1 for index, w in enumerate(ls)}
        src_word_dict['PAD'] = config.PAD
        # 构建id2word映射
        src_index_dict = {v: k for k, v in src_word_dict.items()}
        return src_word_dict, src_vocab_size, src_index_dict

    def build_tgt_dict(self, sentences, max_words=5e4):
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加PAD,BOS,EOS
        ls = word_count.most_common(int(max_words))
        tgt_vocab_size = len(ls) + 3
        tgt_word_dict = {w[0]: index + 3 for index, w in enumerate(ls)}
        tgt_word_dict['PAD'] = 0
        tgt_word_dict['BOS'] = 1
        tgt_word_dict['EOS'] = 2
        # 构建id2word映射
        tgt_index_dict = {v: k for k, v in tgt_word_dict.items()}
        return tgt_word_dict, tgt_vocab_size, tgt_index_dict

    def __getitem__(self, index):
        # 单词映射为索引
        enc_input_tokens = [self.src_word_dict[word] for word in self.data_src[index]]
        dec_input_tokens = [self.tgt_word_dict[word] for word in self.data_tgt[index]]

        enc_num_padding_tokens = self.max_seq_len - len(enc_input_tokens) - 2 # BOS EOS
        dec_num_padding_tokens = self.max_seq_len - len(dec_input_tokens) - 1 # BOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.BOS,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.EOS,
                self.PAD.repeat(enc_num_padding_tokens),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.BOS,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.PAD.repeat(dec_num_padding_tokens),
            ],
            dim=0,
        )

        decoder_output = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.EOS,
                self.PAD.repeat(dec_num_padding_tokens),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert decoder_output.size(0) == self.max_seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.PAD).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.PAD).unsqueeze(0).int() & causal_mask(
                decoder_output.size(0)),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": decoder_output,  # (seq_len)
            "src_text": self.data_src[index] + [" "] * (enc_num_padding_tokens + 2),
            "tgt_text": self.data_tgt[index] + [" "] * (dec_num_padding_tokens + 1),
        }
        # mask中False是需要被忽略掉的，True是需要被保留的

    def __len__(self):
        return len(self.data_src)


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_dataloader(datasets, batch_size, num_workers, ratio):
    # 计算训练集和验证集的大小
    total_size = len(datasets)
    train_size = int(total_size * ratio)
    val_size = total_size - train_size

    # 使用 random_split 将数据集划分为训练集和验证集
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    # 创建训练集的数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 训练数据集
        batch_size=batch_size,  # 每个batch的大小
        shuffle=True,  # 是否在每个epoch开始时打乱数据
        num_workers=num_workers,  # 使用多少个子进程来加载数据
    )

    # 创建验证集的数据加载器
    val_dataloader = DataLoader(
        dataset=val_dataset,  # 验证数据集
        batch_size=1,  # 每个batch的大小
        shuffle=False,  # 验证阶段通常不需要打乱数据
        num_workers=num_workers,  # 使用多少个子进程来加载数据
    )

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    dataset = MyDatasets(config.TRAIN_FILE, max_seq_len=30)
    x = dataset[0]
    print(x)
    # length = dataset.__len__()
    # for i in range(length):
    #     x, y, z = dataset[i]
    #     if not (type(x) == type(y) == type(z)):
    #         print(f"Index {i}: Types of x, y, z are different.")
    #         print(f"x: {x} (type: {type(x)})")
    #         print(f"y: {y} (type: {type(y)})")
    #         print(f"z: {z} (type: {type(z)})")

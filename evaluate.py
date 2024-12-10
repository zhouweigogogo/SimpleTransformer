import math
from collections import Counter
from torchmetrics.text.bleu import BLEUScore
 
# from torchmetrics.text import BLEUScore
# preds = ['the cat is on the mat']
# target = [['there is a cat on the mat', 'a cat is on the mat']]
# bleu = BLEUScore()
# score = bleu(preds, target)
# print(score)

from torchmetrics.text import BLEUScore

# 确保预测和目标句子都是小写，并且没有额外的标点符号
# preds = ['但愿我是一个好歌手。']
# target = [['但愿我是一个好歌手。']] # 注意这里已经没有额外的标点符号并且是小写
preds = ['但 愿 我']
target = [['但 愿 我']]
# 创建 BLEUScore 的实例
bleu = BLEUScore(n_gram=3)

# 计算 BLEU 分数
score = bleu(preds, target)

# 打印分数
print(score)
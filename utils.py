from load_data import causal_mask
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def greedy_decode(model, source, source_mask, tgt_word_dict, max_len):
    sos_idx = tgt_word_dict["BOS"]
    eos_idx = tgt_word_dict["EOS"]

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).cuda()
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).cuda()

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).cuda()], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def set_seed(seed_value=42):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss_curve(x_list, train_list, val_list, save_path):
    """
    绘制训练和验证的损失曲线并保存到指定路径。
    """

    plt.figure()
    
    # 绘制训练损失曲线，用蓝色表示
    plt.plot(x_list, train_list, marker='o', label='Train Loss', color='blue')
    
    # 绘制验证损失曲线，用红色表示
    plt.plot(x_list, val_list, marker='s', label=' Val  Loss', color='red')
    
    # 设置图表标题和坐标轴标签
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # 添加网格
    plt.grid(True)
    
    # 显示图例
    plt.legend()
    
    # 保存图形到指定路径
    plt.savefig(save_path)
    
    # 关闭图形以释放内存
    plt.close()

    print(f"Loss curves have been saved to {save_path}")
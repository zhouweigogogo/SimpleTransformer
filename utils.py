from a_load_data import causal_mask
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import config
from torch.optim.lr_scheduler import LambdaLR
import math

def greedy_decode(model, source, source_mask, tgt_word_dict, max_len, device):
    bos_idx = tgt_word_dict["BOS"]
    eos_idx = tgt_word_dict["EOS"]

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def beam_search_decode(model, beam_size, source, source_mask, tgt_word_dict, max_len, device):
    bos_idx = tgt_word_dict["BOS"]
    eos_idx = tgt_word_dict["EOS"]

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(source).to(device)

    candidates = [(decoder_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()

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

def seq_padding(X, padding=config.PAD):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def warmup_opt(optimizer, num_warmup_epochs, num_training_epochs, warmup_init_lr=1e-5, warmup_target_lr=1e-3, min_lr=1e-5, last_epoch=-1):
    """
    创建一个线性学习率调度器，该调度器将在`num_warmup_epochs`个epochs内从`init_lr`线性增加到`target_lr`，
    然后保持`target_lr`直到`num_training_epochs`。
    """

    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # 在warmup阶段线性增加学习率
            return float(warmup_init_lr + (warmup_target_lr - warmup_init_lr) * current_epoch / num_warmup_epochs) / warmup_target_lr
        else:
            # Warmup之后保持目标学习率不变
            cosine_progress = (current_epoch - num_warmup_epochs) / (num_training_epochs - num_warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_progress))
            # 确保学习率不低于min_lr
            min_lr_factor = min_lr / warmup_target_lr
            return max(cosine_decay, min_lr_factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def plot_lr_curve(x_list, lr_list, save_path):
    """
    绘制学习率曲线并保存到指定路径。
    """

    plt.figure()
    
    # 绘制验证损失曲线，用红色表示
    plt.plot(x_list, lr_list, marker='s', label='LR  Loss', color='red')
    
    # 设置图表标题和坐标轴标签
    plt.title('LR Curves')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    
    # 添加网格
    plt.grid(True)
    

    # 保存图形到指定路径
    plt.savefig(save_path)
    
    # 关闭图形以释放内存
    plt.close()

    print(f"LR curves have been saved to {save_path}")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def noam_lr_scheduler(d_model, warmup_steps, factor):
    def lr_lambda(step_num):
        if step_num == 0:
            step_num = 1
        return factor * (d_model ** (-0.5)) * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))
    
    return lr_lambda
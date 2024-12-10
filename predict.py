from load_data import MyDatasets, get_dataloader
import config
from model.model import Transformer
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
import csv
from utils import set_seed, greedy_decode

if not os.path.exists(config.SAVE_PATH):
    os.makedirs(config.SAVE_PATH)

set_seed(42)

datasets = MyDatasets(data_path=config.DATA_FILE, max_seq_len=config.MAX_LENGTH)

train_dataloader, val_dataloader = get_dataloader(datasets=datasets, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, ratio=0.9)

# 加载模型
if os.path.exists(os.path.join(config.SAVE_PATH, "best.pt")):
    model = torch.load(os.path.join(config.SAVE_PATH, "best.pt"))
    model.eval()  

    with open(os.path.join(config.SAVE_PATH, 'translate.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        file_heads = ['src_text', 'tgt_text', 'model_out_text']
        writer = csv.DictWriter(csvfile, fieldnames=file_heads)
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch["encoder_input"].cuda() # (b, seq_len)
                encoder_mask = batch["encoder_mask"].cuda() # (b, 1, 1, seq_len)

                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = greedy_decode(model, encoder_input, encoder_mask,datasets.tgt_word_dict, config.MAX_LENGTH)
                model_out_text = "".join([datasets.tgt_index_dict[w.item()] for w in model_out])

                src_text = " ".join([i[0] for i in batch['src_text']])
                tgt_text = "".join([i[0][0] for i in batch['tgt_text']])
                model_out_text = model_out_text.replace('BOS', '').replace('EOS', '')


                # print(src_text)
                # print(tgt_text)
                # print(model_out_text)

                writer.writerow({'src_text': src_text, 'tgt_text': tgt_text, 'model_out_text': model_out_text})
                break



    
from a_load_data import MyDatasets, get_dataloader
import config
from model.model import Transformer
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
import csv
from utils import set_seed, greedy_decode, beam_search_decode
import torchmetrics


if not os.path.exists(config.SAVE_PATH):
    os.makedirs(config.SAVE_PATH)

set_seed(42)

datasets = MyDatasets(data_path=config.DATA_FILE, max_seq_len=config.MAX_LENGTH)

train_dataloader, val_dataloader = get_dataloader(datasets=datasets, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, ratio=0.9)

# 加载模型
if os.path.exists(os.path.join(config.SAVE_PATH, "best.pt")):
    model = torch.load(os.path.join(config.SAVE_PATH, "best.pt"))
    model.eval()  

    tgt_texts, predict_texts = [], []

    with open(os.path.join(config.SAVE_PATH, 'translate.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        file_heads = ['src_text', 'tgt_text', 'model_out_text']
        writer = csv.writer(csvfile)
        writer.writerow(file_heads)
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch["encoder_input"].cuda() # (b, seq_len)
                encoder_mask = batch["encoder_mask"].cuda() # (b, 1, 1, seq_len)

                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = beam_search_decode(model,beam_size=2, source=encoder_input, source_mask=encoder_mask,tgt_word_dict=datasets.tgt_word_dict, max_len=config.MAX_LENGTH, device=config.DEVICE)
                # model_out = greedy_decode(model, source=encoder_input, source_mask=encoder_mask,tgt_word_dict=datasets.tgt_word_dict, max_len=config.MAX_LENGTH, device=config.DEVICE)
                model_out_text = "".join([datasets.tgt_index_dict[w.item()] for w in model_out])

                src_text = " ".join([i[0] for i in batch['src_text']])
                tgt_text = "".join([i[0][0] for i in batch['tgt_text']])
                model_out_text = model_out_text.replace('BOS', '').replace('EOS', '')


                # print("src_text:", src_text)
                # print("tgt_text:", tgt_text.strip())
                # print("model_out_text:", model_out_text.strip())

                tgt_texts.append([" ".join([i[0][0] for i in batch['tgt_text'] if i[0][0] != ' '])])
                predict_texts.append(" ".join([datasets.tgt_index_dict[w.item()] for w in model_out if w.item() not in [1,2]]))

                writer.writerow([src_text.strip(), tgt_text.strip(), model_out_text.strip()])
                # break
        csvfile.close()
        # metric1 = torchmetrics.CharErrorRate()
        # cer = metric1(predict_texts, tgt_texts)

        # metric2 = torchmetrics.WordErrorRate()
        # wer = metric2(predict_texts, tgt_texts)
        print(tgt_texts)
        print(predict_texts)
        metric3 = torchmetrics.BLEUScore(n_gram=3)
        bleu = metric3(predict_texts, tgt_texts)
        print(bleu)



    
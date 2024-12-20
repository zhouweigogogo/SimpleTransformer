from b_load_data_warmup import MyDatasets, get_dataloader
import config
from model.model_prenorm import Transformer
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
from utils import set_seed, warmup_opt, plot_loss_curve, plot_lr_curve, get_lr

if not os.path.exists(config.SAVE_PATH):
    os.makedirs(config.SAVE_PATH)

set_seed(42)

datasets = MyDatasets(data_path=config.DATA_FILE, max_seq_len=config.MAX_LENGTH)

train_dataloader, val_dataloader = get_dataloader(datasets=datasets, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, ratio=0.9)

model = Transformer(src_vocab_size=datasets.src_vocab_size, tgt_vocab_size=datasets.tgt_vocab_size, d_model=config.D_MODEL, d_ffn=config.D_FFN, N=config.N,
                    heads=config.HEADS,
                    dropout_prob=config.DROPOUT_PROB).cuda()

# optimizer = optim.SGD(model.parameters(), lr=config.LR)
optimizer = optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.98), eps=1e-9)
scheduler = warmup_opt(optimizer, num_warmup_epochs=config.WARMUP, num_training_epochs=config.EPOCHS, warmup_init_lr=config.WARMUP_INIT_LR, warmup_target_lr=config.WARMUP_TARGET_LR, min_lr=config.MIN_LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

min_loss = 1e5
all_train_loss, all_val_loss = [], []
lr_list = []

for epoch in range(config.EPOCHS):
    model.train()
    epoch_train_loss = 0
    with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{config.EPOCHS} Train', unit=' batch') as train_pbar:
        for batch in train_dataloader:
            encoder_input = batch['encoder_input'].cuda()  # (B, seq_len)
            decoder_input = batch['decoder_input'].cuda()  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].cuda()  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].cuda()  # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                        decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            label = batch['label'].cuda()  # (B, seq_len)

            loss = loss_fn(proj_output.view(-1, datasets.tgt_vocab_size), label.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_pbar.update(1)
            current_lr = get_lr(optimizer)
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.6f}')

            epoch_train_loss += loss.item()
        all_train_loss.append(epoch_train_loss / len(train_dataloader))

        scheduler.step() # 更新学习率
        lr_list.append(scheduler.get_last_lr()[0])

    model.eval()
    val_loss = 0
    epoch_val_loss = 0
    with tqdm(total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{config.EPOCHS}  Val ', unit=' batch') as val_pbar:
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch['encoder_input'].cuda()  # (B, seq_len)
                decoder_input = batch['decoder_input'].cuda()  # (B, seq_len)
                encoder_mask = batch['encoder_mask'].cuda()  # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].cuda()  # (B, 1, seq_len, seq_len)

                encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                            decoder_mask)  # (B, seq_len, d_model)
                proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

                label = batch['label'].cuda()  # (B, seq_len)

                loss = loss_fn(proj_output.view(-1, datasets.tgt_vocab_size), label.view(-1))
                val_loss += loss.item()

                val_pbar.update(1)
                val_pbar.set_postfix(loss=f'{loss.item():.4f}')

                epoch_val_loss += loss.item()
            all_val_loss.append(epoch_val_loss / len(val_dataloader))
        
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model, os.path.join(config.SAVE_PATH, "best.pt"))

    torch.save(model, os.path.join(config.SAVE_PATH, "last.pt"))

plot_loss_curve(x_list=[i for i in range(config.EPOCHS)], train_list=all_train_loss, val_list=all_val_loss, save_path=os.path.join(config.SAVE_PATH, "loss.png"))
plot_lr_curve(x_list=[i for i in range(config.EPOCHS)], lr_list=lr_list, save_path=os.path.join(config.SAVE_PATH, "lr.png"))
from load_data import MyDatasets, get_dataloader, causal_mask
import config
from model.model import Transformer
import torch.nn as nn
import torch.optim as optim
import torch


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


datasets = MyDatasets(config.TRAIN_FILE, max_seq_len=config.MAX_LENGTH)

train_dataloader, val_dataloader = get_dataloader(datasets, config.BATCH_SIZE, config.NUM_WORKERS, 0.8)

model = Transformer(datasets.src_vocab_size, datasets.tgt_vocab_size, config.D_MODEL, config.D_FFN, config.N,
                    config.HEADS,
                    config.DROPOUT_PROB).cuda()
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=config.LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

for epoch in range(config.EPOCHS):
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
        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"Epoch {epoch}: loss: {loss.item():.2f}")

    model.eval()

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].cuda()  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].cuda()  # (b, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask,
                                      datasets.tgt_word_dict, config.MAX_LENGTH)
            model_out_text = "".join([datasets.tgt_index_dict[w.item()] for w in model_out])

            print(batch["src_text"])
            print(batch["tgt_text"])
            print(model_out_text)

            break

    torch.save(model, 'MyTransformer.pth')

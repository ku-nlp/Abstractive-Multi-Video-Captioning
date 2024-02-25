import os
from typing import List

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def count_model_params(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[2]
    tgt_seq_len = tgt.shape[1]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)     # (src_seq_len, src_seq_len)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)         # (tgt_seq_len, tgt_seq_len)

    src_batch_len = src.shape[0]
    src_padding_mask = torch.zeros(src_batch_len, src_seq_len, device=DEVICE).type(torch.bool)
    # True if src[i][:][k][:] is zeros
    src_padding_mask[:, :] = (src[:, :, :] == 0).all(dim=3).all(dim=1)    # (batch_size, src_seq_len)
    tgt_padding_mask = (tgt == PAD_IDX)     # (batch_size, tgt_seq_len)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def create_mask_for_connection(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)     # (src_seq_len, src_seq_len)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)         # (tgt_seq_len, tgt_seq_len)

    src_batch_len = src.shape[0]
    src_padding_mask = torch.zeros(src_batch_len, src_seq_len, device=DEVICE).type(torch.bool)
    # True if src[i][k][:] is zeros
    src_padding_mask[:, :] = (src[:, :] == 0).all(dim=2)    # (batch_size, src_seq_len)
    tgt_padding_mask = (tgt == PAD_IDX)     # (batch_size, tgt_seq_len)
    # print(src_mask.shape, src_padding_mask.shape)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])), dim=0)


# collate_fn (single input)
def collate_fn_single(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tensor_transform(tgt_sample))
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    print(src_batch.shape, tgt_batch.shape)
    return src_batch, tgt_batch


# collate_fn (multiple input)
def collate_fn_multi(batch):
    src_batch, tgt_batch = [], []
    max_len_src = 0
    for src_sample, tgt_sample in batch:
        max_len_src = max(max_len_src, src_sample.shape[1])
        src_batch.append(src_sample)
        tgt_batch.append(tensor_transform(tgt_sample))

    # 系列長を max_len_sec に揃える
    for i in range(len(src_batch)):
        src_sample = src_batch[i]
        seq_len = src_sample.shape[1]
        pad_2d = (0, 0, 0, max_len_src - seq_len)
        src_batch[i] = torch.nn.functional.pad(src_sample, pad_2d, "constant", 0)

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


def train_epoch(tokenizer, model, train_dataloader, optimizer):
    model.train()
    losses = 0

    with tqdm(train_dataloader) as pbar:
        pbar.set_description(f'[train]')
        for src, tgt in pbar:
            # import pdb; pdb.set_trace()

            input_ids = tokenizer(src, return_tensors='pt', padding=True).input_ids.to(DEVICE)
            labels = tokenizer(tgt, return_tensors='pt', padding=True).input_ids.to(DEVICE)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()

            optimizer.step()
            losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(tokenizer, model, val_dataloader):
    model.eval()
    losses = 0

    with tqdm(val_dataloader) as pbar:
        pbar.set_description(f'[eval]')
        for src, tgt in pbar:
            input_ids = tokenizer(src, return_tensors='pt', padding=True).input_ids.to(DEVICE)
            labels = tokenizer(tgt, return_tensors='pt', padding=True).input_ids.to(DEVICE)

            loss = model(input_ids=input_ids, labels=labels).loss
            losses += loss.item()

    return losses / len(val_dataloader)


class EarlyStopping:
    """earlystoppingクラス
    https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503
    """

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数: 最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = float("inf")   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score > self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        model.save_pretrained(self.path)
        print(f'Model saved to {self.path}')
        self.val_loss_min = val_loss  #その時のlossを記録する


if __name__ == "__main__":
    print("utils.py is only for import")

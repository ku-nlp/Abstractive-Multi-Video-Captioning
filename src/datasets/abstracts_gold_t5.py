from email.mime import base
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import fasttext
import fasttext.util


def generate_dataset(**kwargs):
    return Cap2AbsDataset(**kwargs)


class Cap2AbsDataset(Dataset):
    """
    Abstractive Dataset (Gold Caption to Abstractive Caption)
    """
    def __init__(self, jsonl_dir, split_type, num_videos, **kwargs):
        self.jsonl_dir = jsonl_dir
        self.split_type = split_type
        assert self.split_type in ["train", "val", "test"]

        jsonl_path = os.path.join(self.jsonl_dir, self.split_type + ".jsonl")
        with open(jsonl_path, "r") as file:
            jsonl_list = [json.loads(line) for line in file.read().splitlines()]

        json_path_vatex = os.path.join(self.jsonl_dir, "vatex_id2cap.json")
        with open(json_path_vatex, "r") as file:
            self.vatex_dict = json.load(file)

        self.video_ids = []
        self.captions = []
        for data_dict in jsonl_list:
            videos = data_dict["videos"]
            if len(videos) != num_videos:   # 動画数が num_videos のデータだけを使う
                continue
            video_id_list = [videos[i]["video_id"] for i in range(len(videos))]
            self.video_ids.append(video_id_list)
            sentence = data_dict["caption"]
            self.captions.append(sentence)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id_list = self.video_ids[idx]

        caps = []
        for i in range(len(video_id_list)):
            vatex_caption = self.vatex_dict[video_id_list[i]][0]
            assert vatex_caption
            caps.append(vatex_caption)

        # T5に入力するためにフォーマット
        src = 'summarize: '
        src += ' '.join(caps)

        tgt = self.captions[idx]

        return src, tgt


if __name__ == "__main__":
    from make_vocab import make_vocab
    JSONL_DIR = "./data/abstractive_preprocessed"
    feature_dir = "./data/VATEX_features"

    vocab = make_vocab(os.path.join(JSONL_DIR, "train.jsonl"))
    num_videos = 6
    test_dataset = AbstractiveDataset(JSONL_DIR, feature_dir, "test", vocab, num_videos)
    print(test_dataset[-1][0].shape)    # (seq, video_feat_dim x num_videos)
    # for i in range(len(test_dataset)):
    #     tmp = test_dataset[i][0]

    print(len(test_dataset))

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
    return Pred2AbsDataset(**kwargs)


class Pred2AbsDataset(Dataset):
    """
    Abstractive Dataset (soft-aligned features between two videos)
    """
    def __init__(self, jsonl_dir, split_type, vocab, num_videos, feature_dir, **kwargs):
        self.jsonl_dir = jsonl_dir
        self.split_type = split_type
        assert self.split_type in ["train", "val", "test"]
        self.vocab = vocab
        self.feature_dir = feature_dir
        if self.split_type == "test":
            self.caption_type = "pred"
        else:
            self.caption_type = "gold"

        jsonl_path = os.path.join(self.jsonl_dir, self.split_type + ".jsonl")
        with open(jsonl_path, "r") as file:
            jsonl_list = [json.loads(line) for line in file.read().splitlines()]

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

        cap_reps = []
        for i in range(len(video_id_list)):
            video_id = video_id_list[i]
            rep_path = os.path.join(self.feature_dir, self.caption_type, f"{video_id}.npy")
            rep = torch.tensor(np.load(rep_path).squeeze())
            cap_reps.append(rep)
        cap_reps = pad_sequence(cap_reps, batch_first=True)

        sentence = self.captions[idx]
        sequence = [self.vocab[word] for word in sentence.split()]

        return cap_reps, sequence

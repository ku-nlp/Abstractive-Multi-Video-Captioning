import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_dataset(**kwargs):
    return AbstrActsDataset(**kwargs)


class AbstrActsDataset(Dataset):
    def __init__(self, jsonl_dir, feature_dir, split_type, vocab, num_videos, **kwargs):
        self.jsonl_dir = jsonl_dir
        self.feature_dir = os.path.abspath(feature_dir)
        self.split_type = split_type
        assert self.split_type in ["train", "val", "test"]
        self.vocab = vocab
        self.num_videos = num_videos

        jsonl_path = os.path.join(self.jsonl_dir, self.split_type + ".jsonl")
        with open(jsonl_path, "r") as file:
            jsonl_list = [json.loads(line) for line in file.read().splitlines()]

        self.video_id_lists = []
        self.captions = []
        self.ter_score_lists = []
        for data_dict in jsonl_list:
            video_info_list = data_dict["videos"]   # list of dicts
            # if len(video_info_list) != num_videos:   # 動画数が num_videos のデータだけを使う (Optional)
            #     continue
            # video_info_list = sorted(video_info_list, key=lambda x: x["ter_human"], reverse=True)    # sort by TER score
            video_ids = []
            ter_scores = []
            for video_info in video_info_list:
                # if video_info['entailment'] <= 3:   # Seed Caption の TER スコアが 3 以下のデータは使わない (Optional)
                #     continue
                video_ids.append(video_info["video_id"])
                ter_scores.append(video_info["entailment"])
            if len(video_ids) <= 1:
                continue
            self.video_id_lists.append(video_ids)
            self.ter_score_lists.append(ter_scores)
            sentence = data_dict["caption"]          # Human Caption を正解ラベルにする
            # sentence = data_dict["vatex_caption"]    # Seed Caption を正解ラベルにする
            self.captions.append(sentence)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        video_ids = self.video_id_lists[idx]
        if self.split_type in ["train", "val"]:
            split_dir = "trainval"
        elif self.split_type == "test":
            split_dir = "public_test"

        video_reps = []
        for video_id in video_ids:
            rep_path = os.path.join(self.feature_dir, split_dir, f"{video_id}.npy")
            rep = np.load(rep_path).squeeze()
            video_reps.append(rep)
        video_reps = torch.tensor(np.array(video_reps))
        # print(video_reps.size())

        sentence = self.captions[idx]
        sequence = [self.vocab[word] for word in sentence.split()]

        return video_reps, sequence


if __name__ == "__main__":
    from make_vocab import make_vocab
    JSONL_DIR = "./data/abstractive_preprocessed"
    feature_dir = "./data/VATEX_features"

    vocab = make_vocab(os.path.join(JSONL_DIR, "train.jsonl"))
    num_videos = 6
    test_dataset = AbstrActsDataset(JSONL_DIR, feature_dir, "test", vocab, num_videos)
    print(test_dataset[-1][0].shape)    # (seq, video_feat_dim x num_videos)
    # for i in range(len(test_dataset)):
    #     tmp = test_dataset[i][0]

    print(len(test_dataset))

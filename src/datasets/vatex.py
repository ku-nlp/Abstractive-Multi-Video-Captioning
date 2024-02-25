import json
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset


def generate_dataset(**kwargs):
    return VatexDataset(**kwargs)


class PretrainDataset(Dataset):
    """
    VATEX Dataset
    """
    def __init__(self, json_dir, feature_dir, split_type, vocab):
        self.json_dir = json_dir
        self.feature_dir = feature_dir
        self.split_type = split_type
        assert self.split_type in ["train", "val", "test"]
        self.vocab = vocab

        if self.split_type == "train":
            json_path = os.path.join(self.json_dir, "training.jsonl")
            self.split_dir = "trainval"
        elif self.split_type == "val":
            json_path = os.path.join(self.json_dir, "validation.jsonl")
            self.split_dir = "trainval"
        if self.split_type == "test":
            json_path = os.path.join(self.json_dir, "publictest.jsonl")
            self.split_dir = "public_test"

        with open(json_path, "r") as file:
            json_list = [json.loads(line) for line in file.read().splitlines()]

        self.video_ids = []
        self.captions = []
        for data_dict in json_list:
            video_name = data_dict["video_id"]
            video_name += ".npy"
            feature_path = os.path.join(self.feature_dir, self.split_dir, video_name)
            if not os.path.exists(feature_path):
                print(f"video feature not found: {feature_path}")
                continue
            if self.split_type == "train":
                # video and caption pairs
                for caption in data_dict["captions"]:
                    self.video_ids.append(video_name)
                    self.captions.append(caption)
            elif self.split_type in ["val", "test"]:
                # video and first-caption pair
                self.video_ids.append(video_name)
                sentences = data_dict["captions"][0]
                self.captions.append(sentences)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feature_path = os.path.join(self.feature_dir, self.split_dir, video_id)

        video_features = np.load(feature_path).squeeze()
        video_features = np.concatenate((video_features, video_features), axis=1)

        sentence = self.captions[idx]
        sequence = [self.vocab[word] for word in sentence.split()]

        return video_features, sequence


class VatexDataset(Dataset):
    """
    VATEX Dataset
    """
    def __init__(self, json_dir, feature_dir, split_type, vocab, **kwargs):
        self.json_dir = json_dir
        self.feature_dir = feature_dir
        self.split_type = split_type
        assert self.split_type in ["train", "val", "test"]
        self.vocab = vocab

        if self.split_type == "train":
            json_path = os.path.join(self.json_dir, "training.jsonl")
            self.split_dir = "trainval"
        elif self.split_type == "val":
            json_path = os.path.join(self.json_dir, "validation.jsonl")
            self.split_dir = "trainval"
        elif self.split_type == "test":
            json_path = os.path.join(self.json_dir, "publictest.jsonl")
            self.split_dir = "public_test"

        with open(json_path, "r") as file:
            json_list = [json.loads(line) for line in file.read().splitlines()]

        self.video_ids = []
        self.captions = []
        num_lost_features = 0
        for data_dict in json_list:
            video_name = data_dict["video_id"]
            video_name += ".npy"
            feature_path = os.path.join(self.feature_dir, self.split_dir, video_name)
            if not os.path.exists(feature_path):
                num_lost_features += 1
                continue
            if self.split_type == "train":
                # video and caption pairs
                for caption in data_dict["captions"]:
                    self.video_ids.append(video_name)
                    self.captions.append(caption)
            elif self.split_type in ["val", "test"]:
                # video and first-caption pair
                self.video_ids.append(video_name)
                sentences = data_dict["captions"][0]
                self.captions.append(sentences)
        print(f"{num_lost_features} features are not found ({self.split_type})")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feature_path = os.path.join(self.feature_dir, self.split_dir, video_id)

        video_features = np.load(feature_path).squeeze()
        video_features = torch.tensor(np.array([video_features]))
        # print(video_features.size())

        sentence = self.captions[idx]
        sequence = [self.vocab[word] for word in sentence.split()]

        return video_features, sequence


if __name__ == "__main__":
    from make_vocab_vatex_preprocessed import make_vocab
    JSON_DIR = "./data/VATEX_features/preprocessed_jsonl"
    feature_dir = "./data/VATEX_features"

    vocab = make_vocab(os.path.join(JSON_DIR, "training.jsonl"))
    training_dataset = VatexDataset(JSON_DIR, feature_dir, "train", vocab)
    sample_feature, sample_sequence = training_dataset[0]
    print(sample_sequence)
    print(sample_feature.shape)
    print(len(training_dataset))

import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Integration(nn.Module):
    def __init__(self, input_size: int, emb_size):
        super(Integration, self).__init__()
        self.linear = nn.Linear(input_size, emb_size)

    def forward(self, video_reps):
        output = self.linear(video_reps)
        output = torch.mean(output, dim=1)
        return output


class Concatenation(nn.Module):
    def __init__(self, input_size: int, emb_size):
        super(Concatenation, self).__init__()
        self.linear = nn.Linear(input_size, emb_size)

    def forward(self, video_reps):
        video_reps = [video_reps[:, i] for i in range(video_reps.shape[1])]
        output = torch.cat(video_reps, dim=2)
        output = self.linear(output)
        # print(output.shape)
        return output


class Connection(nn.Module):
    def __init__(self, input_size: int, emb_size):
        super(Connection, self).__init__()
        self.linear = nn.Linear(input_size, emb_size)

    def forward(self, video_reps):
        video_reps = [video_reps[:, i] for i in range(video_reps.shape[1])]
        output = torch.cat(video_reps, dim=1)
        output = self.linear(output)
        # print(output.shape)
        return output

class SoftAlignment(nn.Module):
    def __init__(self, input_size: int, emb_size: int):
        super(SoftAlignment, self).__init__()
        self.linear = nn.Linear(input_size, emb_size)

    def calc_similarity(self, rep1, rep2):
        rep2 = rep2.transpose(1, 2)
        # print(rep1.shape, rep2.shape)
        dot_matrix = torch.bmm(rep1, rep2)
        norm1 = torch.linalg.vector_norm(rep1, dim=2)
        norm1 = torch.unsqueeze(norm1, dim=2)
        # print("norm1:", norm1.shape)
        norm2 = torch.linalg.vector_norm(rep2, dim=1)
        norm2 = torch.unsqueeze(norm2, dim=1)
        # print("norm2:", norm2.shape)
        dot_matrix = dot_matrix / norm1
        dot_matrix = dot_matrix / norm2
        dot_matrix = torch.nan_to_num(dot_matrix, 0.0)
        # print("dot_matrix:", dot_matrix.shape)
        return dot_matrix

    def soft_alignment(self, video_reps, ter_scores=None):
        # Shape of video_reps: (batch, num_videos, seq_len, vec_dim)
        num_videos = video_reps.shape[1]
        if ter_scores is not None:
            weights = [ter_scores[i] / sum(ter_scores) for i in range(num_videos)]
        else:
            # weights = [1 / len(video_reps)] * len(video_reps)
            weights = [1] * num_videos
        base_rep = video_reps[:, 0]
        aligned_features = base_rep * weights[0]
        for i in range(1, num_videos):
            if False:       # 動画グループに含まれる動画数が、モデルに入力したい動画数より少ない場合の処理
                aligned_features = np.pad(aligned_features, ((0, 0), (0, base_rep.shape[1])))
                continue
            rep = video_reps[:, i]
            similarity = self.calc_similarity(base_rep, rep)
            next_rep = torch.bmm(similarity, rep) * weights[i]
            aligned_features = torch.cat([aligned_features, next_rep], axis=2)
            # print(aligned_features.shape)
        return aligned_features

    def forward(self, video_reps):
        output = self.soft_alignment(video_reps)
        output = self.linear(output)
        return output

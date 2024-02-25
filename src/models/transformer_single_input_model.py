"""
reference: https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer

from src.models import layers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_model(**kwargs):
    return SingleInputTransformer(**kwargs)


# Seq2Seq Network (with trainable embedding layer)
class SingleInputTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 input_size: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first: bool = True):
        super(SingleInputTransformer, self).__init__()
        self.src_emb = nn.Linear(input_size, emb_size)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=batch_first)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = layers.TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding_src = layers.PositionalEncoding(emb_size, dropout)
        self.positional_encoding_tgt = layers.PositionalEncoding(emb_size, dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding_src(self.src_emb(src))
        tgt_emb = self.positional_encoding_tgt(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding_src(self.src_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding_tgt(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

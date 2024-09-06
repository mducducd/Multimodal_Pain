from model.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from model.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
# from AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
# from Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from torch import nn


class ConvTran(nn.Module):
    '''
    emb_size', type=int, default=16, help='Internal dimension of transformer embeddings'
    dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer'
    num_heads', type=int, default=8, help='Number of multi-headed attention heads'
    '''
    def __init__(self, channel_size, clip_length, emb_size=16, num_heads=8, dim_ff=256, Fix_pos_encode='tAPE', 
                    dropout=0.01, Rel_pos_encode='eRPE', num_classes=3):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = channel_size, clip_length
        emb_size = emb_size
        num_heads = num_heads
        dim_ff = dim_ff
        self.Fix_pos_encode = Fix_pos_encode
        self.Rel_pos_encode = Rel_pos_encode
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=dropout, max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=dropout, max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, dropout)
        else:
            self.attention_layer = Attention(emb_size, num_heads, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, channel_size, clip_length, emb_size=16, num_heads=8, dim_ff=256, Fix_pos_encode='tAPE', 
                    dropout=0.01, Rel_pos_encode='eRPE', num_classes=3):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = channel_size, clip_length
        emb_size = emb_size
        num_heads = num_heads
        dim_ff = dim_ff
        self.Fix_pos_encode = Fix_pos_encode
        self.Rel_pos_encode = Rel_pos_encode
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=dropout, max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, dropout)
        else:
            self.attention_layer = Attention(emb_size, num_heads, dropout)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x)
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out
# import torch
# x = torch.rand([16, 300, 50])
# model = ConvTran(300, 50)
# print(model(x).shape)


from torch.nn.modules.transformer import _get_clones
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def get_tgt_mask(biased_mask, T):
    biased_mask = biased_mask[:, :T, :T].clone().detach()
    mask_last = torch.zeros(T, T)
    mask_last[-1,:-1] = 1
    mask = torch.eye(T, T)
    mask = mask + mask_last
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    biased_mask = mask.unsqueeze(0) + biased_mask
    return biased_mask

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_biased_mask2(n_head, window_size, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))

    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(window_size, max_seq_len)
    for i in range(window_size):
        alibi[i, :max_seq_len - window_size + i +1] = bias[window_size-i-1:]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = torch.triu(torch.ones(window_size, max_seq_len))  == 1
    mask = torch.flip(mask, dims = [0, 1])

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device,  T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, i*2:i*2+2] = 0
    return (mask==1).to(device=device)

def enc_dec_mask2(device, T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)


def enc_dec_mask3(device, T, S):
    mask = torch.ones(T, S)
    # if dataset == "BIWI":
    #     for i in range(T):
    #         mask[i, i*2:i*2+2] = 0
    # elif dataset == "vocaset":
    #     for i in range(T):
    #         mask[i, i] = 0
    for i in range(T):
        mask[i, :S-T+i+1] = 0
    return (mask==1).to(device=device)



# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=751):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class MBT(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_bottle_token, device):
        super(MBT, self).__init__()
        self.dim = dim
        self.device = device
        self.num_layers = num_layers
        self.num_bottle_token = num_bottle_token
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.x_layers = _get_clones(encoder_layer, num_layers)
        self.y_layers = _get_clones(encoder_layer, num_layers)

        self.bot = nn.Parameter(torch.randn(1, num_bottle_token, dim))

    def get_mask(self, b, l):
        return torch.zeros(b, l+self.num_bottle_token).to(device=self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mask_x = self.get_mask(x.shape[0], x.shape[1])
        mask_y = self.get_mask(y.shape[0], y.shape[1])

        bot = self.bot.expand(x.shape[0], -1, -1)
        x = torch.cat((bot, x), dim=1)
        y = torch.cat((bot, y), dim=1)

        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        for i in range(self.num_layers):
            x = self.x_layers[i](src=x, src_key_padding_mask=mask_x)
            y = self.y_layers[i](src=y, src_key_padding_mask=mask_y)

            x[:self.num_bottle_token] = (x[:self.num_bottle_token] + y[:self.num_bottle_token]) / 2
            y[:self.num_bottle_token] = x[:self.num_bottle_token]

        x = x[self.num_bottle_token:,:,:].permute(1, 0, 2)
        y = y[self.num_bottle_token:,:,:].permute(1, 0, 2)

        return x, y
    
class SpeakFormer(nn.Module):
    def __init__(self, img_size=224, feature_dim = 256, period = 25, max_seq_len = 751,  device = 'cpu', use_mbt=True):
        super(SpeakFormer, self).__init__()

        self.use_mbt = use_mbt
        self.img_size = img_size

        self.feature_dim = feature_dim

        self.audio_feature_map = nn.Linear(768, feature_dim)

        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)


        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2*feature_dim, batch_first=True)
        self.speaker_transformer_decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=1)


        self.speaker_transformer_decoder3 = MBT(feature_dim, 2, 4, 4, device)
  

        self.device = device

    def forward(self, video_features, audio):

        frame_num = video_features.shape[1]

        hidden_states = audio

        if hidden_states.shape[1]<frame_num*2:
            video_features = video_features[:, : hidden_states.shape[1]//2]
            frame_num = hidden_states.shape[1]//2

        hidden_states = self.audio_feature_map(hidden_states)

        video_features = self.PPE(video_features)
        # tgt_mask = self.biased_mask[:, :video_features.shape[1], :video_features.shape[1]].clone().detach().to(device=self.device).repeat(video_features.shape[0],1,1)
        # memory_mask = enc_dec_mask(self.device, video_features.shape[1], hidden_states.shape[1])

        # speaker_vector = self.speaker_transformer_decoder1(video_features, video_features, tgt_mask=tgt_mask)
        # speaker_vector = self.speaker_transformer_decoder2(speaker_vector, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)

        speaker_motion, hidden_states = self.speaker_transformer_decoder3(video_features, hidden_states)


        return  speaker_motion


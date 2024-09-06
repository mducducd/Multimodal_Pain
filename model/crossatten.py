import math
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
class DenseCoAttn(nn.Module):

	def __init__(self, dim1, dim2, dropout):
		super(DenseCoAttn, self).__init__()
		dim = dim1 + dim2
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
		self.query_linear = nn.Linear(dim, dim)
		self.key1_linear = nn.Linear(16, 16)
		self.key2_linear = nn.Linear(16, 16)
		self.value1_linear = nn.Linear(dim1, dim1)
		self.value2_linear = nn.Linear(dim2, dim2)
		self.relu = nn.ReLU()

	def forward(self, value1, value2):
		joint = torch.cat((value1, value2), dim=-1)
		# audio  audio*W*joint
		joint = self.query_linear(joint)
		key1 = self.key1_linear(value1.transpose(1, 2))
		key2 = self.key2_linear(value2.transpose(1, 2))
		value1 = self.value1_linear(value1)
		value2 = self.value2_linear(value2)

		weighted1, attn1 = self.qkv_attention(joint, key1, value1, dropout=self.dropouts[0])
		weighted2, attn2 = self.qkv_attention(joint, key2, value2, dropout=self.dropouts[1])


		return weighted1, weighted2

	def qkv_attention(self, query, key, value, dropout=None):
		d_k = query.size(-1)
		scores = torch.bmm(key, query) / math.sqrt(d_k)
		scores = torch.tanh(scores)
		if dropout:
			scores = dropout(scores)

		weighted = torch.tanh(torch.bmm(value, scores))
		return self.relu(weighted), scores
class NormalSubLayer(nn.Module):

    def __init__(self, dim1, dim2, dropout):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(dim1, dim2, dropout)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2):
        weighted1, weighted2 = self.dense_coattn(data1, data2)
        data1 = data1 + self.linears[0](weighted1)
        data2 = data2 + self.linears[1](weighted2)

        return data1, data2


class DCNLayer(nn.Module):

    def __init__(self, dim1, dim2, num_seq, dropout, length1, length2, num_classes):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer(dim1, dim2, dropout) for _ in range(num_seq)])
        self.linear1 = nn.Linear(length1, 16)
        self.linear2 = nn.Linear(length2, 16)
        self.regressor1 = nn.Linear(dim1+dim2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.regressor2 = nn.Linear(256, num_classes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(dim1+dim2, num_classes)

    def forward(self, data1, data2):
        data1 = self.linear1(data1.transpose(1, 2)).transpose(1, 2)
        data2 = self.linear2(data2.transpose(1, 2)).transpose(1, 2)
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2)

        out = torch.cat((data1, data2), dim=-1)
        # print(out.shape)
        # c = self.regressor1(c).transpose(1, 2)
        # c = self.bn1(c).transpose(1, 2)
        # c = F.leaky_relu(c)
        # c = self.regressor2(c)
        # c = torch.tanh(c)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out